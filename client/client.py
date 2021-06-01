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
from utils.memory_monitor import get_memory_used
from utils import logger


class CachedFP32Buff(object):
    # TODO release max_chunk_size
    def __init__(self, default_chunk_size: int, rank: int):
        self.cached_chunk_id = None
        self.max_chunk_size = default_chunk_size
        self.rank = rank
        self.cpu_cached_fp32_payload = torch.zeros(self.max_chunk_size,
                                                   dtype=torch.float,
                                                   pin_memory=True)
        logger.info(f'CachedFP32Buff init with rank {self.rank}')
        self.cuda_cached_fp32_payload = torch.zeros(
            self.max_chunk_size,
            dtype=torch.float,
            device=torch.device(f'cuda:{self.rank}'))

    def reset(self):
        self.cached_chunk_id = None
        self.cpu_cached_fp32_payload.zero_()
        self.cuda_cached_fp32_payload.zero_()

    def update_chunk(self, chunk: Chunk, time_profile=True):
        """
        如果chunk id被cache住，则直接cached_buff上索引
        chunk在cuda上，返回结果再cpu上
        cuda fp16 -> cpu fp16 -> cpu fp32
        cuda fp16 -> cpu fp32 慢！
        """
        chunk_id = chunk.chunk_id
        if self.cached_chunk_id is None or self.cached_chunk_id != chunk_id:
            if time_profile:
                start_time = time.time()

            self.cached_chunk_id = chunk_id
            chunk_size = chunk.capacity
            if chunk_size > self.max_chunk_size:
                self.max_chunk_size = chunk_size
                self.cpu_cached_fp32_payload = torch.zeros(self.max_chunk_size,
                                                           dtype=torch.float,
                                                           pin_memory=True)
                self.cuda_cached_fp32_payload = torch.zeros(
                    self.max_chunk_size,
                    dtype=torch.float,
                    device=torch.device(f'cuda:{self.rank}'))

            cuda_buff = self.cuda_cached_fp32_payload.narrow(0, 0, chunk_size)
            cuda_buff.copy_(chunk.payload)
            cpu_buff = self.cpu_cached_fp32_payload.narrow(0, 0, chunk_size)
            cpu_buff.copy_(cuda_buff)
            # self.cpu_cached_fp32_payload.copy_(chunk.payload)

            if time_profile:
                global_timer.gpu_cpu_move_elapse += time.time() - start_time
                global_timer.gpu_cpu_move_times += 1
                global_timer.gpu_cpu_move_data_amount += chunk.capacity

    def access_chunk(self, start_offset, numel):
        return self.cpu_cached_fp32_payload.narrow(0, start_offset, numel)


class HybridPSClient(object):
    def __init__(self,
                 rank: int = 0,
                 default_chunk_size: int = 1024 * 1024,
                 warmup=True,
                 is_fp16=False):
        """
        管理一个Process的Param, AccGrad, OS数据。
        每个进程可以访问一个GPU的显存，和cpu的内存
        功能:
          1. 充分利用cpu和gpu内存
          2. 细粒度调度，HybridPSClient包含若干chunk
        """
        self.pid = os.getpid()

        # index of gpu
        self.rank = rank

        self.chunk_list = ChunkList()

        self.module = None

        # 解耦chunk和param的tensors
        self.chunk_tensor_index = ChunkTensorIndex()
        self.chunk_schema_scheduler = None
        self.default_chunk_size = default_chunk_size
        self._time_profile = True

        self._is_warmup = warmup
        self._warmup_phase = True
        # 通过运行一次迭代来动态进行chunk schduling
        self._is_fp16 = is_fp16

        self._chunk_id = -1
        self._cached_fp32_buff = CachedFP32Buff(default_chunk_size, rank)

    def _generate_chunk_id(self):
        self._chunk_id += 1
        return self._chunk_id

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

    def unset_warmup_phase(self):
        timer = global_timer.IterationTimer()
        timer.warmup = False
        self.chunk_list.moments_cnt_of_iteration = timer.moment()

    def init(self, model, optimizer):
        self.module = model
        self.optimizer = optimizer

        self.static_chunk_schedule(model, optimizer)

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        # TODO(jiaruifang) 目前仍是调试状态，每个进程有全部模型
        self._copy_model(model)

        self.register_model_hook(model)

        # self.chunk_tensor_index.visit_chunks(self.chunk_list)

    def static_chunk_schedule(self, model, optimizer):
        """
        TODO(jiaruifang)静态调度chunk schema
        注册模型和优化器，相当于静态图的预处理过程
        执行chunk schema调取，为每个tensor找到对应的chunk和位置
        """
        self.chunk_schema_scheduler = ChunkShemaScheduler(
            self.default_chunk_size, model, optimizer, self.chunk_list,
            self.chunk_tensor_index)
        if self._is_fp16:
            self.chunk_schema_scheduler.schedule_fp16()
        else:
            self.chunk_schema_scheduler.schedule()
        logging.info(f"static_chunk_schedule finished")

    def register_model_hook(self, model):
        setup_hybrid_ps_hooks(model, self)

    def _copy_model(self, model):
        # 拷贝模型
        for i, group in enumerate(self.optimizer.param_groups):
            for j, param in enumerate(group['params']):
                self.access_data(param, torch.device('cpu:0'))
                data_tensor = param.ps_attr.access_tensor(AccessType.DATA)
                data_tensor.copy_(param.data)

                if self._is_fp16:
                    param_fp32 = self.optimizer.state[param]['fp32_param_data']
                    self.access_data(param_fp32, torch.device('cpu:0'))
                    data_tensor_fp32 = param_fp32.ps_attr.access_tensor(
                        AccessType.DATA)
                    data_tensor_fp32.copy_(param.data.float())
                    self.release_data(param_fp32, PSTensorStatus.HOLD)

                self.release_data(param, PSTensorStatus.HOLD)

    def generate_grad_params(self):
        """
        生成当前chunk list中所有grad tensors
        """
        return self.chunk_tensor_index.generate_grad_tensor_param()

    def fp16_to_fp32_copy(self, param, access_type):
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.chunk_tensor_index.get_tensor_info(tensor_id)
        self._cached_fp32_buff.update_chunk(self.chunk_list[info.chunk_id])
        return self._cached_fp32_buff.access_chunk(info.start_offset,
                                                   info.numel)

    def _assign_chunk_for_tensor(self, param, access_type):
        """
        为param分配一个chunk，如果已经存在的chunk有空隙则插在空隙中
        如果没有空隙则分配一个新的chunk
        """
        numel = param.ps_attr.ps_numel
        data_type = param.dtype

        chunk_id, offset = self.chunk_tensor_index.find_gap(numel, data_type)

        # 如果没有gap需要新分配一个
        # 还要拷贝数据
        if chunk_id is None:
            chunk_id = self._generate_chunk_id()
            offset = 0
            # logging.info(f"no gap need to new a chunk_id {chunk_id} numel {numel} data type {data_type}")
            chunk_size = max(self.default_chunk_size, numel)
            self.chunk_list.new_chunk(chunk_id, chunk_size, data_type)
            self.chunk_tensor_index.add_chunk(chunk_id, chunk_size, data_type)
        else:
            # logging.info(f"find_gap chunk_id {chunk_id} numel {numel} data type {data_type}")
            pass

        if access_type == AccessType.DATA:
            self.chunk_tensor_index.add_tensor(chunk_id,
                                               param.ps_attr.data_id(), offset,
                                               numel, param, AccessType.DATA)
        elif access_type == AccessType.GRAD:
            self.chunk_tensor_index.add_tensor(chunk_id,
                                               param.ps_attr.grad_id(), offset,
                                               numel, param, AccessType.GRAD)
        return chunk_id

    def access_grad_dist(self, param: torch.nn.Parameter,
                         compute_device: torch.device):
        """
        在单机多卡训练过程中，访问param的grad tensor。
        找到param.grad对应的chunk，获得global chunks。
        等待global chunks都从compute状态变成hold，执行reduce-scatter
        每个
        """
    def access_dist(self, param: torch.nn.Parameter, access_type: AccessType,
                    compute_device: torch.device):
        if self._time_profile:
            start_time = time.time()

        assert compute_device.type == "cuda"
        if not hasattr(param, 'ps_attr'):
            register_param(param)
            raise RuntimeError(
                "FP16 training shall not meet tensors not registered for PS")

        # 准备param所在chunk的内存，如果内存不在计算设备上需要分配或者移动
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        chunk_id_list = self.chunk_tensor_index.get_global_chunk_id_list(
            chunk_id)
        rank = torch.distributed.get_rank()
        local_chunk_id = chunk_id_list[rank]
        logger.info(
            f'rank {rank} access_dist local_chunk_id {local_chunk_id} chunk_id_list {chunk_id_list}'
        )

        # 每个进程把chunk_id_list都弄到本地
        self.chunk_list.access_chunk_dist(local_chunk_id, chunk_id_list,
                                          compute_device)

        # 将param内存定位到chunk上
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.chunk_tensor_index.get_tensor_info(tensor_id)
        start_offset = info.start_offset
        numel = info.numel
        assert numel == param.ps_attr.ps_numel, f"{numel} vs {param.ps_attr.ps_numel}"

        param.ps_attr.set_tensor(
            self.chunk_list[chunk_id].payload.narrow(0, start_offset, numel),
            access_type)

        ### 改变param's tensor对应chunk的status，chunk状态由它管理的所有tensor状态共同决定。
        old_status = param.ps_attr.get_status(access_type)

        # 如果是从free/uninit状态转换的需要清零
        if old_status == PSTensorStatus.FREE or old_status == PSTensorStatus.UNINIT:
            param.ps_attr.access_tensor(access_type).zero_()

        # 访问之后应该更新Tensor的状态，tensor对应的chunk的状态随之改变
        # dist情况
        # 对于global chunk_list其他chunk，他们的payload被分配出来了，他们的status生效
        # 可以被换入换出。如果payload size是0也不会被换入换出。
        # 如果on local则可以被本进程换入换出。
        # TODO release时候要被释放payload
        self.chunk_list.update_status(chunk_id, old_status,
                                      PSTensorStatus.COMPUTE)
        param.ps_attr.set_status(PSTensorStatus.COMPUTE, access_type)

        if self._time_profile:
            global_timer.client_access_elapse += time.time() - start_time

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
        两种access，
        1. FWD+BWD过程的access_data and access_grad?
        如果访问的是本地chunk_id，也需要allgather配合其他process
        2. adam过程access_data
        只能访问本地chunk不需要通信
        """
        if self._time_profile:
            start_time = time.time()

        if not hasattr(param, 'ps_attr'):
            # 第一次access，动态调度方案的预热过程会遇到
            # data 和 grad都有id
            # TODO(jiaruifang)可以在optimizer init过程把编号分配好
            register_param(param)
            raise RuntimeError(
                "FP16 training shall not meet tensors not registered for PS")

        # 准备param所在chunk的内存，如果内存不在计算设备上需要分配或者移动
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        # 这个tensor还没在chunk schema中
        is_first_init = False
        if chunk_id is None:
            raise RuntimeError(
                "FP16 training shall not meet tensors with no chunk assigned")
            chunk_id = self._assign_chunk_for_tensor(param, access_type)
            # logging.info(f'not found chunk, assign chunk {chunk_id} access type {access_type} {param.dtype}')
            is_first_init = True
        else:
            pass
            # logging.info(f'found chunk_id {chunk_id}  access type {access_type} {param.dtype}')

        rank = torch.distributed.get_rank()
        logger.debug(
            f'rank {rank} accesses chunk id {chunk_id} payload {self.chunk_list[chunk_id].payload}'
        )

        self.chunk_list.access_chunk(chunk_id, compute_device)
        # 将param内存定位到chunk上
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.chunk_tensor_index.get_tensor_info(tensor_id)
        start_offset = info.start_offset
        numel = info.numel
        assert numel == param.ps_attr.ps_numel, f"{numel} vs {param.ps_attr.ps_numel}"

        param.ps_attr.set_tensor(
            self.chunk_list[chunk_id].payload.narrow(0, start_offset, numel),
            access_type)

        old_status = param.ps_attr.get_status(access_type)

        # 如果是从free状态转换的需要清零，或者从
        if old_status == PSTensorStatus.FREE or old_status == PSTensorStatus.UNINIT:
            param.ps_attr.access_tensor(access_type).zero_()

        # 第一次分配ps data时候要把原来param的tensor拷贝过来
        if is_first_init:
            if access_type == AccessType.DATA:
                param.ps_attr.access_tensor(access_type).copy_(param.data)
            elif access_type == AccessType.GRAD and param.grad is not None:
                param.ps_attr.access_tensor(access_type).copy_(param.grad)

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
            self.access(param, AccessType.DATA, compute_device)
            chunk_id = param.ps_attr.ps_data_chunk_id
            # TODO(jiarufiang) 需要记录device信息
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
        # if param.ps_attr.get_status(access_type) != PSTensorStatus.COMPUTE:
        #     return

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
            # TODO(jiaruifang) 必须to device它和param.grad_fn.next_functions[0][0]
            param.data = torch.zeros(1,
                                     dtype=param.dtype,
                                     device=torch.device('cpu:0')).to(
                                         param.device)
        elif access_type == AccessType.GRAD:
            param.grad = None

        # PS：主要耗时部分，如果真正执行，以下代码非常耗时。可以改成分配时再释放。
        # 不应该删除chunks，因为fp16的chunk可以被复用。fp32 chunk不存在删除情况
        self.chunk_list.delete_free_chunks()
        # TODO(jiaruifang) remote chunk且是free需要释放payload

        if self._time_profile:
            global_timer.client_release_elapse += time.time() - start_time

    def release_data(self,
                     param: torch.nn.Parameter,
                     reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        """
        可以把一个tensor释放成FREE，也可以成HOLD
        """
        timer = global_timer.IterationTimer()
        self.release(param, AccessType.DATA, reset_to_status)

    def release_grad(self,
                     param: torch.nn.Parameter,
                     reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        timer = global_timer.IterationTimer()
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
