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

from .chunk_data import Chunk
from .chunk_list import ChunkList
from .chunk_tensor_index import ChunkTensorIndex
from .const import PSTensorStatus, ChunkListType
from .parameter import PSParameter, register_param, is_param_registed, is_torch_param, register_torch_param
import torch
import logging
import sys
from patrickstar.utils import logger
from typing import List


class ChunkCreator(object):
    def __init__(self, default_chunk_size: int, chunk_list: ChunkList,
                 chunk_tensor_index: ChunkTensorIndex,
                 dummy_param_list: List[torch.nn.Parameter]):
        """
        更新chunk_list和chunk_tensor_index，来建立chunk-tensor schema
        """
        # default chunk size是subchunk size
        self.default_chunk_size = default_chunk_size
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index

        self.chunk_id = 0
        self.acc_cnt = 0
        self.data_type = None

        self.world_size = 1
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()

        # fp16 tensors, fp32 tensors, m tensors, v tensors各组成一个list
        # list_id表示tensor在list的顺序。global id跨list需要清零
        self.list_id = 0
        self.global_chunk_id = 0

        self.dummy_param_list = dummy_param_list
        logger.info(f'default chunk size {default_chunk_size}')

    def _add_tensor(self, tensor_id, numel, param,
                    data_type, chunk_list_type: ChunkListType):
        """
        向chunk_tensor_index注册tensor，当新的Tensor大小超过Chunk剩余空间，
        则开辟一个新的Chunk添加到ChunkList中，将Tensor分配在新Chunk的起始位置
        返回chunk在chunk group中的位置
        """
        # data_type甚至可以和param不一致
        self.data_type = data_type
        if self.acc_cnt + numel > self.default_chunk_size:
            # 如果再加入一个tensor就超过default_chunk_size了，将已经积累的tensor打包
            self.chunk_list.new_chunk(self.chunk_id, self.default_chunk_size,
                                      self.data_type)
            self.chunk_tensor_index.add_chunk(self.chunk_id,
                                              self.default_chunk_size,
                                              self.data_type,
                                              self.global_chunk_id,
                                              chunk_list_type)
            # 标记chunk在global chunk中的位置
            self.list_id += 1
            if self.list_id % self.world_size == 0:
                self.global_chunk_id += 1

            # 为下一个chunk准备
            self.chunk_id += 1
            self.acc_cnt = 0

        self.chunk_tensor_index.add_tensor(self.chunk_id, tensor_id,
                                           self.acc_cnt, numel, param)
        self.acc_cnt += numel

        # 返回tensor对应的chunk在chunk group的位置
        return (self.list_id) % self.world_size, self.chunk_id

    def _start_new_chunk_list(self, add_dummy_chunk_flag: bool,
                              chunk_list_type: ChunkListType):
        """
        对构造中的chunk进行收尾
        """
        if self.acc_cnt > 0:
            self.chunk_list.new_chunk(self.chunk_id, self.default_chunk_size,
                                      self.data_type)
            self.chunk_tensor_index.add_chunk(self.chunk_id,
                                              self.default_chunk_size,
                                              self.data_type,
                                              self.global_chunk_id,
                                              chunk_list_type)
            self.chunk_id += 1
            self.acc_cnt = 0
            # 下一个chunk的list_id
            self.list_id += 1

        # 给不足world_size的global chunk补上dummy chunk，每个dummy chunk管理一个dummy param
        if add_dummy_chunk_flag:
            while self.list_id % self.world_size != 0:
                logger.info('add dummy chunk')
                self.chunk_list.new_chunk(self.chunk_id,
                                          self.default_chunk_size,
                                          self.data_type,
                                          is_dummy=True)
                self.chunk_tensor_index.add_chunk(self.chunk_id,
                                                  self.default_chunk_size,
                                                  self.data_type,
                                                  self.global_chunk_id,
                                                  chunk_list_type)
                dummy = torch.nn.Parameter(torch.tensor([], dtype=self.data_type),
                                           requires_grad=False)
                # 加入一个dummy param可以让dummy chunk状态被设置为hold
                register_param(dummy, "dummy")
                self.dummy_param_list.append(dummy)
                self.chunk_tensor_index.add_tensor(
                    self.chunk_id, self.dummy_param_list[-1].ps_attr.id(),
                    0, 1, self.dummy_param_list[-1])

                self.chunk_id += 1
                self.list_id += 1

        self.list_id = 0
        self.global_chunk_id += 1


class ChunkShemaScheduler(object):
    def __init__(self, default_chunk_size, chunk_list,
                 chunk_tensor_index: ChunkTensorIndex):
        self.default_chunk_size = default_chunk_size
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index
        self.dummy_param_list = []

        self.chunk_creator = ChunkCreator(default_chunk_size, chunk_list,
                                          chunk_tensor_index,
                                          self.dummy_param_list)

    def add_tensor(self, tensor_id, numel, param,
                   data_type, chunk_list_type):
        """
        返回chunk在process chunk group中的位置
        """
        return self.chunk_creator._add_tensor(tensor_id, numel, param,
                                              data_type, chunk_list_type)

    def start_new_chunk_list(self, add_dummy_chunk_flag: bool,
                             chunk_list_type: ChunkListType):
        self.chunk_creator._start_new_chunk_list(add_dummy_chunk_flag,
                                                 chunk_list_type)

    def schedule(self, module, optimizer):
        """
        为module和optimizer的参数指定chunk schema
        schedule过程为所有parameter注册成ps_tensor
        """
        # 注册model和optimizer的param是否应该和chunk layout schedule耦合？
        # TODO(jiaruifang)用一次FWD来得到正确的执行顺序
        # FP16 和 FP32不一样
        raise NotImplementedError
        for name, param in module.named_parameters(recurse=True):
            register_param(param, name)
            numel = param.numel()
            data_type = torch.float
            self.add_tensor(param.ps_attr.id(), numel, param,
                            data_type, ChunkListType.PARAM_FP32)

        self.chunk_creator.start_new_chunk_list(True, ChunkListType.PARAM_FP32)

        # layout for M, V
        # Parameters`tate[param_group][param]`尚未被初始化，它们在第一次step时候才出现
        # 这里提前计算它们的layout，因此需要将它们提前弄出来
        # TODO(jiaruifang) 不是FWD计算顺序，是init初始化顺序
        for group in optimizer.param_groups:
            for p in group['params']:
                # TODO(jiaruifang)应该执行一次backward计算，然后才能知道p的grad是否为None，不能用requires_grad来代替？
                if p.requires_grad is True:

                    state = optimizer.state[p]
                    # Eager state initialization, different from Pytorch
                    if len(state) == 0:
                        state['step'] = 0
                        # 被PatrickStar管理
                        # Exponential moving average of gradient values
                        data_type = p.dtype
                        state['exp_avg'] = torch.nn.Parameter(
                            torch.zeros(
                                p.ps_attr.shape,
                                dtype=data_type,
                                # memory_format=torch.preserve_format,
                                device=torch.device('cpu:0')),
                            requires_grad=False)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.nn.Parameter(
                            torch.zeros(
                                p.ps_attr.shape,
                                dtype=data_type,
                                # memory_format=torch.preserve_format,
                                device=torch.device('cpu:0')),
                            requires_grad=False)

                        ps_name_prefix = p.ps_attr.name
                        register_param(state['exp_avg'],
                                       f'{ps_name_prefix}.exp_avg')
                        register_param(state['exp_avg_sq'],
                                       f'{ps_name_prefix}.exp_avg_sq')

                        numel = p.ps_attr.numel
                        self.add_tensor(state['exp_avg'].ps_attr.id(),
                                        numel, state['exp_avg'],
                                        data_type, ChunkListType.VARIANCE)

                        self.add_tensor(state['exp_avg_sq'].ps_attr.id(),
                                        numel, state['exp_avg_sq'],
                                        data_type, ChunkListType.MOMENTUM)

                        # param.data不被需要，将他们的内存释放
                        state['exp_avg'].data = torch.tensor(
                            [], dtype=data_type, device=torch.device('cpu:0'))
                        state['exp_avg_sq'].data = torch.tensor(
                            [], dtype=data_type, device=torch.device('cpu:0'))

                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            raise NotImplementedError
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)
                else:
                    raise RuntimeError
        self.chunk_creator.start_new_chunk_list(False, ChunkListType.MOMENTUM)

    def schedule_fp16(self, module, optimizer):
        """
        为module和optimizer的参数指定chunk schema
        schedule过程为所有parameter注册成ps_tensor
        顺便统计param fp16有几个chunk
        """
        self.start_new_chunk_list(add_dummy_chunk_flag=True,
                                  chunk_list_type=ChunkListType.PARAM_FP16)

        # Note: param_fp16_chunk_num包括dummy chunk
        self.chunk_tensor_index.param_fp16_chunk_num = self.chunk_tensor_index.get_cur_chunk_num()

        # 注册param data fp32，只有属于local chunk的param才初始化内存
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if not is_torch_param(p):
                    param_fp32 = state['fp32_param_data']
                    numel = param_fp32.ps_attr.numel
                    chunk_pos = self.add_tensor(param_fp32.ps_attr.id(),
                                                numel, param_fp32, torch.float, ChunkListType.PARAM_FP32)

        self.start_new_chunk_list(add_dummy_chunk_flag=False,
                                  chunk_list_type=ChunkListType.PARAM_FP32)

        # 注册M，V, param data fp32
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = optimizer.state[p]
                    # Exponential moving average of gradient values
                    if not is_torch_param(p):
                        numel = p.ps_attr.numel

                        chunk_pos = self.add_tensor(
                            state['exp_avg'].ps_attr.id(), numel,
                            state['exp_avg'], torch.float, ChunkListType.VARIANCE)

                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            raise NotImplementedError
                else:
                    raise RuntimeError

        self.start_new_chunk_list(add_dummy_chunk_flag=False,
                                  chunk_list_type=ChunkListType.VARIANCE)

        # 注册exp_avg_sq
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad:
                    state = optimizer.state[p]
                    # Exponential moving average of gradient values
                    if not is_torch_param(p):
                        numel = p.ps_attr.numel
                        self.add_tensor(state['exp_avg_sq'].ps_attr.id(),
                                        numel, state['exp_avg_sq'],
                                        torch.float, ChunkListType.MOMENTUM)

                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            raise NotImplementedError
                else:
                    raise RuntimeError

        # 收尾
        self.start_new_chunk_list(add_dummy_chunk_flag=False,
                                  chunk_list_type=ChunkListType.MOMENTUM)

        logging.info('static schedule with FP16 finished')
