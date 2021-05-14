# Copyright (C) 2021 THL A29 Limited, a Tencent company.
# All rights reserved.
# Licensed under the BSD 3-Clause License (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at
# https://opensource.org/licenses/BSD-3-Clause
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
# See the AUTHORS file for names of contributors.

import torch
import os
from manager import HybridPSManager
from typing import Dict
import datetime
import logging
from torch.multiprocessing import Process, Manager

from .hook import setup_hybrid_ps_hooks
from .const import AccessType, PSChunkStatus, PSTensorStatus
from .chunk_data import Chunk
from .chunk_list import ChunkList
from .helper import getsizeof
from .chunk_tensor_index import ChunkTensorIndex
from .chunk_schema_scheduler import ChunkShemaScheduler
import utils.global_timer as global_timer
import time
from .parameter import PSParameter, register_param, is_param_registed
from utils.memory_monitor import get_free_memory


class WarmupInfo(object):
    def __init__(self, chunk_size):
        # 下次分配的chunk offset和chunk id
        self._chunk_offset = 0
        self._chunk_id = 0
        # chunk size刚好大于_defualt_chunk_size
        self._defualt_chunk_size = chunk_size

    def update(self, numel, chunk_list: ChunkList, data_type):
        """
        更新下一个chunk分配地址
        """
        if self._chunk_offset >= self._defualt_chunk_size:
            # 需要下一个chunk了，先把这个chunk分配出来
            chunk_list.new_chunk(self._chunk_id, self._chunk_offset, data_type)
            self._chunk_offset = 0
            self._chunk_id += 1
        else:
            self._chunk_offset += numel


class HybridPSClient(object):
    def __init__(self,
                 gpu_index: int = 0,
                 default_chunk_size: int = 1024 * 1024,
                 warmup=True):
        """
        管理一个Process的Param, AccGrad, OS数据。
        每个进程可以访问一个GPU的显存，和cpu的内存
        功能:
          1. 充分利用cpu和gpu内存
          2. 细粒度调度，HybridPSClient包含若干chunk
        """
        self.pid = os.getpid()

        # index of gpu
        self.gpu_index = gpu_index

        self.chunk_list = ChunkList()

        self.module = None

        # 解耦chunk和param的tensors
        self.chunk_tensor_index = ChunkTensorIndex()
        self.chunk_schema_scheduler = None
        self.default_chunk_size = default_chunk_size
        self._time_profile = True

        self._is_warmup = warmup
        self._warmup_phase = True
        self._warmup_chunk_info = WarmupInfo(self.default_chunk_size)
        # 通过运行一次迭代来动态进行chunk schduling
        self._dynamic_chunk_scheduler = False

    def pre_iter(self):
        if self._is_warmup:
            self.set_warmup_phase()
        timer = global_timer.IterationTimer()
        timer.reset()

    def post_iter(self):
        if self._is_warmup:
            self._is_warmup = False
            self.unset_warmup_phase()

    def set_warmup_phase(self):
        timer = global_timer.IterationTimer()
        timer.warmup = True
        self._warmup_chunk_info = WarmupInfo(self.default_chunk_size)

    def unset_warmup_phase(self):
        self._warmup_chunk_info = WarmupInfo(self.default_chunk_size)
        timer = global_timer.IterationTimer()
        timer.warmup = False
        self.chunk_list.moments_cnt_of_iteration = timer.moment()

    def init(self, model, optimizer):
        self.static_chunk_schedule(model, optimizer)
        self.register_model_hook(model)
        self.copy_model(model)
        self.copy_optimizer(optimizer)
        self.module = model

        # self.chunk_tensor_index.visit_chunks(self.chunk_list)

    def static_chunk_schedule(self, model, optimizer):
        """
        TODO(jiaruifang)静态调度chunk schema
        注册模型和优化器，相当于静态图的预处理过程
        执行chunk schema调取，为每个tensor找到对应的chunk和位置
        """
        assert self._dynamic_chunk_scheduler is False
        self.chunk_schema_scheduler = ChunkShemaScheduler(
            self.default_chunk_size, model, optimizer, self.chunk_list,
            self.chunk_tensor_index)
        self.chunk_schema_scheduler.schedule()
        logging.info(f"static_chunk_schedule finished")

    def register_model_hook(self, model):
        setup_hybrid_ps_hooks(model, self)

    def copy_model(self, model):
        # 拷贝模型
        for name, param in model.named_parameters():
            # TODO(jiaruifang)设备应该是自适应的
            self.access_data(param, torch.device('cpu:0'))
            data_tensor = param.ps_attr.access_tensor(AccessType.DATA)
            data_tensor.copy_(param.data)
            self.release_data(param, PSTensorStatus.HOLD)

    def copy_optimizer(self, optimizer):
        # FP16 master model copy
        if hasattr(optimizer, 'fp32_from_fp16_groups'):
            for param_group in optimizer.fp32_from_fp16_groups:
                for master_param in param_group:
                    self.access_data(master_param, torch.device('cpu:0'))
                    master_param.ps_attr.access_tensor(AccessType.DATA).copy_(
                        master_param.data)
                    self.release_data(master_param, PSTensorStatus.HOLD)

        # inspect chunk layout if you want.
        # self.chunk_tensor_index.visit_chunks(self.chunk_list)

    def access(self, param: torch.nn.Parameter, access_type: AccessType,
               compute_device: torch.device):
        """
        访问`nn.Parameter`的data或者grad，让它们参与计算。
        具体步骤
        找到param的tensor对应的chunk。
        1. 如果chunk的payload存在
        然后决定是否移动chunk到计算设备，移动之前要给计算设备腾出足够空间。
        2. 如果chunkd的payload不存在
        比如grad FP16，在step过程所在的chunk已经被标记为FREE，并被释放掉。
        将payload分配到计算设备的内存上，分配前需要给计算设备腾出足够空间。

        异常：一个chunk中两个tensor的计算设备不一致。
        """
        if self._time_profile:
            start_time = time.time()

        if not hasattr(param, 'ps_attr'):
            raise RuntimeError("access a param without ps_attr")

        # 准备param所在chunk的内存，如果内存不在计算设备上需要分配或者移动
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        self.chunk_list.access_chunk(chunk_id, compute_device)
        # 将param内存定位到chunk上
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.chunk_tensor_index.get_tensor_info(tensor_id)
        start_offset = info.start_offset
        numel = info.numel
        assert numel == param.ps_attr.ps_numel

        param.ps_attr.set_tensor(
            self.chunk_list[chunk_id].payload.narrow(0, start_offset, numel),
            access_type)

        old_status = param.ps_attr.get_status(access_type)

        # 如果是从free状态转换的需要清零
        if old_status == PSTensorStatus.FREE:
            param.ps_attr.access_tensor(access_type).zero_()

        # 访问之后应该更新Tensor的状态，chunk的状态随之改变
        self.chunk_list.update_status(chunk_id, old_status,
                                      PSTensorStatus.COMPUTE)
        param.ps_attr.set_status(PSTensorStatus.COMPUTE, access_type)

        # Note并不设置parameter对应的tensor，因为adam可能直接访问pstensor
        if self._time_profile:
            global_timer.client_access_elapse += time.time() - start_time

    def access_data(self, param: torch.nn.Parameter,
                    compute_device: torch.device):
        """
        将param的ps_data_tensor的数据放置到compute_device上
        """
        timer = global_timer.IterationTimer()
        if timer.warmup:
            # TODO(jiaruifang) 按照tensor在一次迭代第一次出现位置分配chunk
            if self._dynamic_chunk_scheduler:
                # warmup过程首次访问该param
                if not is_param_registed(param):
                    register_param(param)
                numel = param.numel()
                data_type = param.dtype
                chunk_id = self._warmup_chunk_info._chunk_id
                offset = self._warmup_chunk_info._chunk_offset
                self.chunk_tensor_index.add_tensor(chunk_id,
                                                   param.ps_attr.data_id(),
                                                   offset, numel, param,
                                                   AccessType.DATA)
                self._warmup_chunk_info.update(numel, self.chunk_list,
                                               data_type)
                # 赋值一个dummy data
                param.data = torch.zeros(param.ps_attr.ps_shape,
                                         dtype=param.dtype,
                                         device=compute_device)
            # 更新chunk的访问时间
            else:
                self.access(param, AccessType.DATA, compute_device)
            chunk_id = param.ps_attr.ps_data_chunk_id
            self.chunk_list[chunk_id].add_moment(timer.moment())
        else:
            self.access(param, AccessType.DATA, compute_device)

    def access_grad(self, param: torch.nn.Parameter,
                    compute_device: torch.device):
        """
        将param的ps_grad_tensor的数据放置到compute_device上
        NOTE，并没有正确设置param的grad，此时grad的数据无效。因为grad的设备属性并不自由，需要看data的脸色行事。我们使用grad时候，需要显式设置
        `param.grad = param.ps_grad_tensore`
        """
        timer = global_timer.IterationTimer()
        if timer.warmup:
            # 预热阶段 1. 注册参数 2. 分配chunk 3. 添加chunk被访问的信息
            if self._dynamic_chunk_scheduler:
                if not is_param_registed(param):
                    register_param(param)
                    numel = param.numel()
                    data_type = param.dtype
                    chunk_id = self._warmup_chunk_info._chunk_id
                    offset = self._warmup_chunk_info._chunk_offset
                    self.chunk_tensor_index.add_tensor(chunk_id,
                                                       param.ps_attr.grad_id(),
                                                       offset, numel, param,
                                                       AccessType.GRAD)
                    self._warmup_chunk_info.update(numel, self.chunk_list,
                                                   data_type)
                    # 赋值一个dummy grad
                    param.grad = torch.zeros(param.ps_attr.ps_shape,
                                             dtype=param.dtype,
                                             device=compute_device)
            else:
                self.access(param, AccessType.GRAD, compute_device)
            # 更新chunk的访问时间
            chunk_id = param.ps_attr.ps_grad_chunk_id
            self.chunk_list[chunk_id].add_moment(timer.moment())
        else:
            self.access(param, AccessType.GRAD, compute_device)

    def release(self,
                param: torch.nn.Parameter,
                access_type: AccessType,
                reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        """
        这个param的data, grad不再需要放在计算设备
        1. 更新状态
        首先更新tensor和chunk的状态
        2. 释放内存
        在释放Parameter中tensor的内存，释放PSTensor中的内存
        看看是否有chunk的状态为free，释放chunk内存
        """
        if self._time_profile:
            start_time = time.time()

        assert isinstance(reset_to_status, PSTensorStatus)
        if param.ps_attr.get_status(access_type) != PSTensorStatus.COMPUTE:
            return
        # assert param.ps_attr.get_status(
        #     access_type
        # ) == PSTensorStatus.COMPUTE, "param to be released is not at COMPUTE status"

        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        logging.debug(
            f'release {access_type} chunk_id {chunk_id} to {reset_to_status}')

        # 更新tensor和chunk状态， tensor被设置为free，需要删除内存
        # 释放tensor的内存，再释放chunk内存
        self.chunk_list.update_status(chunk_id,
                                      param.ps_attr.get_status(access_type),
                                      reset_to_status)
        param.ps_attr.set_status(reset_to_status, access_type)

        # 找到需要删除的chunk，先删除chunk关联的tensors
        if access_type == AccessType.DATA:
            param.data = torch.zeros(1,
                                     dtype=param.dtype,
                                     device=torch.device('cpu:0'))
        elif access_type == AccessType.GRAD:
            param.grad = None

        if self._time_profile:
            sub_start_time = time.time()

        # 主要耗时部分, TODO(jiaruifang) lazy release
        self.chunk_list.delete_free_chunks()

        if self._time_profile:
            global_timer.memory_delete_elapse += time.time() - sub_start_time

        if self._time_profile:
            global_timer.client_release_elapse += time.time() - start_time

    def release_data(self,
                     param: torch.nn.Parameter,
                     reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        """
        可以把一个tensor释放成FREE，也可以成HOLD
        """
        timer = global_timer.IterationTimer()
        if timer.warmup and self._dynamic_chunk_scheduler:
            param.data = torch.zeros(1, device=torch.device('cpu:0'))
        else:
            self.release(param, AccessType.DATA, reset_to_status)

    def release_grad(self,
                     param: torch.nn.Parameter,
                     reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        timer = global_timer.IterationTimer()
        if timer.warmup and self._dynamic_chunk_scheduler:
            param.grad = None
        else:
            self.release(param, AccessType.GRAD, reset_to_status)

    def reset(self):
        """
        删除chunk_list和chunk_tensor_index
        """
        raise NotImplementedError

    def visit(self):
        for idx, chunk in self.chunk_list.generate_chunk():
            logging.info(
                f"chunk {idx} on device {chunk.device} status {chunk.get_status()}"
            )
            chunk.visit(self.chunk_tensor_index)

    def release_all_data_grad(self, status):
        if self.module is not None:
            for n, p in self.module.named_parameters():
                self.release_grad(p, status)
                self.release_data(p, status)

    def allreduce(self, param, access_type):
        """
        必须所有process同时执行，规约后的payload存储在哪(cpu or gpu)由调度器决定
        """
        pass

    def broadcast(self, param, access_type):
        """
        必须所有process同时执行，规约后的payload存储在哪由调度器决定
        """
        pass
