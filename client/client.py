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
from typing import Dict
import datetime
import logging
from torch.multiprocessing import Process, Manager

from .hook import setup_hybrid_ps_hooks
from .const import AccessType, PSChunkStatus, PSTensorStatus, TrainingStage
from .chunk_data import Chunk
from .chunk_list import ChunkList
from .helper import getsizeof
from .chunk_tensor_index import ChunkTensorIndex
from .chunk_schema_scheduler import ChunkShemaScheduler
import utils.global_timer as global_timer
import time
from .parameter import PSParameter, register_param, is_param_registed, is_torch_param
from utils.memory_monitor import get_sys_memory_used
from utils import logger
from deepspeed_helper.global_vars import get_args
from manager import PatrickStarManager


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


class PatrickStarClient(object):
    def __init__(self,
                 rank: int,
                 default_chunk_size: int,
                 warmup=True,
                 is_fp16=False):
        """
        管理一个Process的Param, AccGrad, OS数据。
        每个进程可以访问一个GPU的显存，和cpu的内存
        功能:
          1. 充分利用cpu和gpu内存
          2. 细粒度调度，PatrickStarClient包含若干chunk
        """
        self.pid = os.getpid()

        # index of gpu
        self.rank = rank

        self.chunk_list = ChunkList()

        self.module = None

        # 解耦chunk和param的tensors
        self.chunk_tensor_index = ChunkTensorIndex()
        self.chunk_schema_scheduler = ChunkShemaScheduler(
            default_chunk_size, self.chunk_list, self.chunk_tensor_index)
        self.default_chunk_size = default_chunk_size
        self._time_profile = True

        # 通过运行一次迭代来动态进行chunk schduling
        self._is_fp16 = is_fp16

        self._chunk_id = -1
        self._cached_fp32_buff = CachedFP32Buff(default_chunk_size, rank)

        self.cpu_comm_group = torch.distributed.new_group(backend='gloo')

    def _generate_chunk_id(self):
        self._chunk_id += 1
        return self._chunk_id

    def init(self, model, optimizer):
        """
        初始化client管理的model和optimizer
        并完成model的参数的分配和拷贝
        """
        self.module = model
        self.optimizer = optimizer
        self.static_chunk_schedule(model, optimizer)

        # self.chunk_tensor_index.visit_chunks(self.chunk_list)

        # TODO(jiaruifang) 目前仍是调试状态，每个进程有全部模型
        self._copy_model(model)

        self.register_model_hook(model)

    def static_chunk_schedule(self, model, optimizer):
        """
        TODO(jiaruifang)静态调度chunk schema
        注册模型和优化器，相当于静态图的预处理过程
        执行chunk schema调取，为每个tensor找到对应的chunk和位置
        """
        if self._is_fp16:
            self.chunk_schema_scheduler.schedule_fp16(model, optimizer)
        else:
            self.chunk_schema_scheduler.schedule(model, optimizer)
        logging.info(f"static_chunk_schedule finished")

    def set_all_tensors_status_in_chunk(self, chunk_id, new_status):
        """
        把一个chunk所有的tensor状态设置为status，chunk的状态也随之改变
        不管payload是否被分配
        """
        chunk = self.chunk_list[chunk_id]
        for info in self.chunk_tensor_index.generate_tensor_info_in_order(
                chunk_id):
            param = info.param
            access_type = info.access_type
            old_status = param.ps_attr.get_status(access_type)
            self.chunk_list.update_status(chunk_id, old_status, new_status)
            param.ps_attr.set_status(new_status, access_type)

    def register_model_hook(self, model):
        setup_hybrid_ps_hooks(model, self)

    def _copy_model(self, model):
        # TODO(jiaruifang)模型参数的初始化顺序和如下循环访问的顺序相同。
        args = get_args()
        for i, group in enumerate(self.optimizer.param_groups):
            for j, param in enumerate(group['params']):
                if is_torch_param(param):
                    if args.cpu_embedding_fp32:
                        self.optimizer.state[param]['fp32_param_data'].copy_(
                            param.data)
                    else:
                        param.data = param.data.half()
                        self.optimizer.state[param]['fp32_param_data'].copy_(
                            param.data.float())
                    continue
                if self.is_local_tensor(param, AccessType.DATA):
                    if True:
                        param.data = param.data.half()
                        self.access_data(param, torch.device('cpu:0'))
                        data_tensor = param.ps_attr.access_tensor(
                            AccessType.DATA)
                        data_tensor.copy_(param.data)
                        if self._is_fp16:
                            param_fp32 = self.optimizer.state[param][
                                'fp32_param_data']
                            self.access_data(param_fp32, torch.device('cpu:0'))
                            data_tensor_fp32 = param_fp32.ps_attr.access_tensor(
                                AccessType.DATA)
                            data_tensor_fp32.copy_(param.data.float())
                            self.release_data(param_fp32, PSTensorStatus.HOLD)
                        self.release_data(param, PSTensorStatus.HOLD)
                    else:
                        if self._is_fp16:
                            param_fp32 = self.optimizer.state[param][
                                'fp32_param_data']
                            self.access_data(param_fp32, torch.device('cpu:0'))
                            data_tensor_fp32 = param_fp32.ps_attr.access_tensor(
                                AccessType.DATA)
                            data_tensor_fp32.copy_(param.data)
                            self.release_data(param_fp32, PSTensorStatus.HOLD)

                        param.data = param.data.half()
                        self.access_data(param, torch.device('cpu:0'))
                        data_tensor = param.ps_attr.access_tensor(
                            AccessType.DATA)
                        data_tensor.copy_(param.data)
                        self.release_data(param, PSTensorStatus.HOLD)
                else:
                    param.data = torch.zeros(1,
                                             dtype=torch.half,
                                             device=param.device)

        for param in self.chunk_schema_scheduler.dummy_param_list:
            if self.is_local_tensor(param, AccessType.DATA):
                self.access_data(param, torch.device('cpu:0'))
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
            chunk_size = max(self.default_chunk_size, numel)
            self.chunk_list.new_chunk(chunk_id, chunk_size, data_type)
            self.chunk_tensor_index.add_chunk(chunk_id, chunk_size, data_type)
        else:
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

    def is_local_tensor(self, param, access_type) -> bool:
        """
        调用本接口前判断是否是torch_param
        判断tensor是否在本GPU之上
        """
        # TODO(jiaruifang)不应该进入这个分支
        if is_torch_param(param):
            return False
        # 准备param所在chunk的内存，如果内存不在计算设备上需要分配或者移动
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        chunk_id_list = self.chunk_tensor_index.get_global_chunk_id_list(
            chunk_id)
        rank = torch.distributed.get_rank()
        assert rank < len(chunk_id_list)
        local_chunk_id = chunk_id_list[rank]
        return chunk_id == local_chunk_id

    def _fetch_remote_chunks(self, chunk_id_list, local_chunk_id,
                             compute_device, param_name,
                             training_stage: TrainingStage):
        """
        将chunk_id_list中远端的chunk取到本地
        """
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # FWD过程，当global chunk中有param第一次被访问时，需要将global chunk收集到本地。
        # 如何判断global chunk中有param第一次被访问的时刻，从而正确触发allgather操作。
        # 第一个param被访问时的必要条件是remote chunk状态为RELEASED。
        # 因此，当每个chunk由HOLD_AFTER_FWD(HOLD_ADFTER_BWD)->RELEASED时
        has_released_chunk = False
        for i in chunk_id_list:
            if self.chunk_list[i].get_status() == PSChunkStatus.RELEASED:
                has_released_chunk = True
                break
        if not has_released_chunk:
            return

        logger.debug(
            f'rank {rank} fetch {param_name} remote chunks {chunk_id_list} local chunk {local_chunk_id}'
        )
        allgather_payload_buff = []

        local_chunk_payload = None
        for chunk_id in chunk_id_list:
            if chunk_id == local_chunk_id:
                local_chunk_payload = self.chunk_list[chunk_id].payload
                allgather_payload_buff.append(local_chunk_payload)
            else:
                self.chunk_list.prepare_device(
                    compute_device,
                    self.chunk_list[chunk_id].get_chunk_space())
                self.chunk_list[chunk_id].allocate_payload(compute_device)
                # 刚分配的chunk，以备allgather使用，allgather之前不要被换出。
                self.chunk_list[chunk_id].pin()
                self.set_all_tensors_status_in_chunk(chunk_id,
                                                     PSTensorStatus.HOLD)
                allgather_payload_buff.append(
                    self.chunk_list[chunk_id].payload)

        for chunk_id in chunk_id_list:
            self.chunk_list[chunk_id].unpin()

        assert torch.distributed.is_initialized(
        ), "torch distributed is not initialized during allgather"

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()

        # TODO(jiaruifang) 部分process做allreduce，用dummy chunk！
        group_list = []
        for i, _ in enumerate(chunk_id_list):
            group_list.append(i)
        group = torch.distributed.new_group(group_list)

        logger.debug(f'rank {rank} allgather {chunk_id_list}')
        handle = torch.distributed.all_gather(allgather_payload_buff,
                                              local_chunk_payload,
                                              group=group,
                                              async_op=False)

        allgather_payload_buff = []

    def access_dist(self, param: torch.nn.Parameter, access_type: AccessType,
                    compute_device: torch.device,
                    training_stage: TrainingStage):
        if self._time_profile:
            start_time = time.time()

        assert compute_device.type == "cuda"
        if not hasattr(param, 'ps_attr'):
            raise RuntimeError(
                "FP16 training shall not meet tensors not registered for PS")

        # 如果是Torch管理的Tensor，则直接返回
        if is_torch_param(param):
            # if access_type == AccessType.DATA:
            #     param.data.to(compute_device)
            # elif access_type == AccessType.GRAD:
            #     param.data.to(compute_device)
            #     param.grad.to(compute_device)
            return
        # 准备param所在chunk的内存，如果内存不在计算设备上需要分配或者移动
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        chunk_id_list = self.chunk_tensor_index.get_global_chunk_id_list(
            chunk_id)
        rank = torch.distributed.get_rank()

        # if rank >= len(chunk_id_list):
        #     return
        assert rank < len(chunk_id_list), f"rank {rank} < {len(chunk_id_list)}"

        local_chunk_id = chunk_id_list[rank]

        logger.debug(
            f'rank {rank} access_dist access tensor {param.ps_attr.ps_name} local_chunk_id {local_chunk_id} chunk_id_list {chunk_id_list}'
        )

        # 每个进程把local_chunk_id都弄到本地
        self.chunk_list.access_chunk(local_chunk_id, compute_device)

        # _fetch_remote_chunks不要将local_chunk_id也给换出去了，因为它的状态还是HOLD，加上pin。
        self.chunk_list[local_chunk_id].pin()

        self._fetch_remote_chunks(chunk_id_list, local_chunk_id,
                                  compute_device, param.ps_attr.ps_name,
                                  training_stage)
        self.chunk_list[local_chunk_id].unpin()

        # _fetch_remote_chunks可能不执行allgather，此时远端的chunk在本地，需要取到计算设备上。
        self.chunk_list.access_chunk(chunk_id, compute_device)

        # 将param内存定位到chunk上
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.chunk_tensor_index.get_tensor_info(tensor_id)
        start_offset = info.start_offset
        numel = info.numel
        assert numel == param.ps_attr.ps_numel, f"{numel} vs {param.ps_attr.ps_numel}"

        assert self.chunk_list[
            chunk_id].payload is not None, f"rank {rank} chunk id {chunk_id}' payload is None'"
        assert self.chunk_list[
            chunk_id].payload.device == compute_device, f"rank {rank} chunk id {chunk_id}' payload is not on {compute_device}, but on {self.chunk_list[chunk_id].payload.device}"

        param.ps_attr.set_tensor(
            self.chunk_list[chunk_id].payload.narrow(0, start_offset, numel),
            access_type)

        ### 改变param's tensor对应chunk的status，chunk状态由它管理的所有tensor状态共同决定。
        old_status = param.ps_attr.get_status(access_type)

        # 如果是从free/uninit状态转换的需要清零
        if old_status == PSTensorStatus.FREE:
            param.ps_attr.access_tensor(access_type).zero_()

        # 访问之后应该更新Tensor的状态，鉴于chunk状态是由它管理tensor共同决定
        # 因此tensor对应的chunk的状态随之改变
        # dist情况
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

        if is_torch_param(param):
            return

        if not hasattr(param, 'ps_attr'):
            # 第一次access，动态调度方案的预热过程会遇到
            # data 和 grad都有id
            # TODO(jiaruifang)可以在optimizer init过程把编号分配好
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
            is_first_init = True
        else:
            pass
        rank = torch.distributed.get_rank()

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
        if old_status == PSTensorStatus.FREE:
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
        self.access(param, AccessType.DATA, compute_device)

    def access_grad(self, param: torch.nn.Parameter,
                    compute_device: torch.device):
        """
        将param的ps_grad_tensor的数据放置到compute_device上
        NOTE，并没有正确设置param的grad，此时grad的数据无效。因为grad的设备属性并不自由，需要看data的脸色行事。我们使用grad时候，需要显式设置
        `param.grad = param.ps_grad_tensore`
        """
        self.access(param, AccessType.GRAD, compute_device)

    def release_dist(self, param: torch.nn.Parameter, access_type: AccessType,
                     reset_to_status: PSTensorStatus,
                     training_stage: TrainingStage, is_allreduce: bool):
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
        rank = torch.distributed.get_rank()

        assert isinstance(reset_to_status, PSTensorStatus)
        assert torch.distributed.is_initialized()

        if is_torch_param(param):
            return

        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        # 可以在tensor-chunk schema构造过程中获得local_chunk_id
        chunk_id_list = self.chunk_tensor_index.get_global_chunk_id_list(
            chunk_id)

        local_chunk_id = chunk_id_list[rank]

        logging.debug(
            f'rank {rank} release tensor {param.ps_attr.ps_name} of chunk_id {chunk_id} to {reset_to_status}'
        )

        # 更新tensor和chunk状态， tensor被设置为free，需要删除内存
        # 释放tensor的内存，再释放chunk内存
        self.chunk_list.update_status(chunk_id,
                                      param.ps_attr.get_status(access_type),
                                      reset_to_status)
        param.ps_attr.set_status(reset_to_status, access_type)

        # 找到需要删除的chunk，先删除chunk关联的tensors
        if access_type == AccessType.DATA:
            # NOTE(jiaruifang) 必须device和原来param一直，影响hook of param.grad_fn.next_functions[0][0]
            param.data = torch.zeros(1, dtype=param.dtype, device=param.device)
        elif access_type == AccessType.GRAD:
            param.grad = None

        # 判断chunk group中所有的chunk都被使用完毕，可以释放remote chunk
        # FWD: 当所有非dummy的chunk都是HOLD_AFTER_FWD
        # BWD: 当所有非dummy的chunk都是HOLD_AFTER_BWD
        all_chunks_ready = True
        for i in chunk_id_list:
            if training_stage == TrainingStage.FWD:
                if not self.chunk_list[i].all_tensor_status(
                        PSTensorStatus.HOLD_AFTER_FWD
                ) and not self.chunk_list[i].is_dummy():
                    all_chunks_ready = False
            elif training_stage == TrainingStage.BWD:
                if not self.chunk_list[i].all_tensor_status(
                        PSTensorStatus.HOLD_AFTER_BWD
                ) and not self.chunk_list[i].is_dummy():
                    all_chunks_ready = False
                # self.chunk_tensor_index.visit_chunk(self.chunk_list[i])
            else:
                raise RuntimeError(
                    f"{training_stage} is neither TrainingStage.FWD nor TrainingStage.BWD"
                )

        if all_chunks_ready:
            if is_allreduce:
                world_size = torch.distributed.get_world_size()
                assert self.chunk_list[local_chunk_id].payload is not None

                group_list = []
                for i, _ in enumerate(chunk_id_list):
                    group_list.append(i)
                group = torch.distributed.new_group(group_list)

                args = get_args()
                if not args.use_fake_dist:
                    input_list = []
                    for i in chunk_id_list:
                        self.chunk_list.access_chunk(
                            i, torch.device(f'cuda:{args.local_rank}'))
                        self.chunk_list[i].pin()
                        input_list.append(self.chunk_list[i].payload)
                    torch.distributed.reduce_scatter(
                        self.chunk_list[local_chunk_id].payload,
                        input_list,
                        op=torch.distributed.ReduceOp.SUM,
                        group=group,
                        async_op=False)
                else:
                    # Note()为了在开发机调试方便
                    # 它非常慢 4.92 sec/5.89 sec (client_release_elapse)
                    temp_start_time = time.time()
                    for rank_, chunk_id_ in enumerate(chunk_id_list):
                        if self.chunk_list[chunk_id_].is_dummy():
                            continue
                        torch.distributed.reduce(
                            self.chunk_list[chunk_id_].payload,
                            rank_,
                            op=torch.distributed.ReduceOp.SUM,
                            group=group,
                            async_op=False)
                    global_timer.temp_check_elapse += time.time(
                    ) - temp_start_time
                # NOTE把下面行注释了不影响最终结果？loss可能是有softmax算出，所以相对值不影响LOSS比较，但是影响了
                # 不应该除以world_size,减去dummy chunk个数
                self.chunk_list[local_chunk_id].payload /= world_size

            # 删除remote chunk的payload
            for i in chunk_id_list:
                self.chunk_list[i].unpin()
                if i != local_chunk_id:
                    logger.debug(f'rank {rank} remove payload of chunk_id {i}')
                    self.chunk_list[i].release_payload()
                    self.set_all_tensors_status_in_chunk(
                        i, PSTensorStatus.FREE)

        if self._time_profile:
            global_timer.client_release_elapse += time.time() - start_time

    def release(self,
                param: torch.nn.Parameter,
                access_type: AccessType,
                reset_to_status: PSTensorStatus = PSTensorStatus.HOLD,
                allreduce_local_grad: bool = False,
                remove_remote_data: bool = False):
        """
        @allreduce_local_grad: 在分布式训练中，对param的tensor进行allreduce
        @remove_remote_data: 在分布式训练中，删除param tensor的payload
        这个param的data, grad不再需要放在计算设备
        1. 更新状态
        首先更新tensor和chunk的状态
        2. 释放内存
        在释放Parameter中tensor的内存，释放PSTensor中的内存
        看看是否有chunk的状态为free，释放chunk内存
        """
        if self._time_profile:
            start_time = time.time()
            # tmp_start_time = time.time()

        if is_torch_param(param):
            return

        args = get_args()
        rank = args.local_rank
        assert isinstance(reset_to_status, PSTensorStatus)

        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        logger.debug(
            f'rank {rank} release a tensor of {access_type} chunk_id {chunk_id} to {reset_to_status}'
        )

        # 更新tensor和chunk状态，如果tensor被设置为free，需要删除ps_tensor的内存
        self.chunk_list.update_status(chunk_id,
                                      param.ps_attr.get_status(access_type),
                                      reset_to_status)
        param.ps_attr.set_status(reset_to_status, access_type)

        # 找到需要删除的chunk，先删除chunk关联的tensors
        if access_type == AccessType.DATA:
            # NOTE() 必须to device它和param.grad_fn.next_functions[0][0]
            param.data = torch.zeros(1, dtype=param.dtype, device=param.device)
        elif access_type == AccessType.GRAD:
            param.grad = None

        if remove_remote_data:
            # FWD逻辑，如果chunk计算完毕非本地的chunk被释放
            if not self.is_local_tensor(
                    param,
                    access_type) and self.chunk_list[chunk_id].get_status(
                    ) == PSChunkStatus.HOLD_AFTER_FWD:
                logger.debug(
                    f'rank {rank} fwd remove payload of chunk_id {chunk_id}')
                self.chunk_list[chunk_id].release_payload()
                self.set_all_tensors_status_in_chunk(chunk_id,
                                                     PSTensorStatus.FREE)
        if allreduce_local_grad:
            # debug分支，一个DPP等价版本，每个chunk在BWD时候同步
            # Chunk状态从compute->HOLD_AFTER_BWD被触发
            if self.chunk_list[chunk_id].get_status(
            ) == PSChunkStatus.HOLD_AFTER_BWD:
                world_size = torch.distributed.get_world_size()
                assert self.chunk_list[chunk_id].payload is not None
                torch.distributed.all_reduce(self.chunk_list[chunk_id].payload,
                                             op=torch.distributed.ReduceOp.SUM,
                                             async_op=False)
                self.chunk_list[chunk_id].payload /= world_size

        if self._time_profile:
            # global_timer.temp_check_elapse += time.time() - tmp_start_time
            global_timer.client_release_elapse += time.time() - start_time

    def release_data(self,
                     param: torch.nn.Parameter,
                     reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        """
        可以把一个tensor释放成FREE，也可以成HOLD
        """
        self.release(param, AccessType.DATA, reset_to_status)

    def release_grad(self,
                     param: torch.nn.Parameter,
                     reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        self.release(param, AccessType.GRAD, reset_to_status)

    def reset(self):
        """
        删除chunk_list和chunk_tensor_index
        """
        raise NotImplementedError

    def visit(self):
        rank = torch.distributed.get_rank()
        for idx, chunk in self.chunk_list.generate_chunk():
            logging.info(
                f"rank {rank} chunk {idx} on device {chunk.get_device()} status {chunk.get_status()}"
            )
            # chunk.visit(self.chunk_tensor_index)
