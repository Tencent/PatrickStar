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
from .const import AccessType, PSTensorStatus
from .parameter import PSParameter, register_param
import torch
import logging
import sys
from utils import logger


class ChunkCreator(object):
    def __init__(self, default_chunk_size: int, chunk_list: ChunkList,
                 chunk_tensor_index: ChunkTensorIndex):
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
        logger.info(f'default chunk size {default_chunk_size}')

    def add_tensor(self, tensor_id, numel, param, access_type: AccessType,
                   data_type):
        """
        向chunk_tensor_index注册tensor，如果超过已有chunk size则新建一个chunk
        相邻add_tensor的data type都是相同的
        所有chunk size都是相同的
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
                                              self.global_chunk_id)
            # 标记chunk在global chunk中的位置
            self.list_id += 1
            if self.list_id % self.world_size == 0:
                self.global_chunk_id += 1

            # 为下一个chunk准备
            self.chunk_id += 1
            self.acc_cnt = 0

        self.chunk_tensor_index.add_tensor(self.chunk_id, tensor_id,
                                           self.acc_cnt, numel, param,
                                           access_type)
        self.acc_cnt += numel

    def start_new_chunk_list(self):
        """
        对构造中的chunk进行收尾
        """
        if self.acc_cnt > 0:
            self.chunk_list.new_chunk(self.chunk_id, self.default_chunk_size,
                                      self.data_type)
            self.chunk_tensor_index.add_chunk(self.chunk_id,
                                              self.default_chunk_size,
                                              self.data_type,
                                              self.global_chunk_id)
            self.chunk_id += 1
            self.acc_cnt = 0

            self.process_id = 0

        self.list_id = 0
        self.global_chunk_id += 1


class ChunkShemaScheduler(object):
    def __init__(self, default_chunk_size, module, optimizer, chunk_list,
                 chunk_tensor_index: ChunkTensorIndex):
        self.module = module
        self.optimizer = optimizer
        self.default_chunk_size = default_chunk_size
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index

        self.chunk_creator = ChunkCreator(default_chunk_size, chunk_list,
                                          chunk_tensor_index)

    def schedule(self):
        """
        为module和optimizer的参数指定chunk schema
        schedule过程为所有parameter注册成ps_tensor
        """
        # 模型参数的data和grad间隔排列。
        # Optimizer的M，V间隔排列
        acc_cnt = 0
        chunk_id = 0

        # 注册model和optimizer的param是否应该和chunk layout schedule耦合？
        # TODO(jiaruifang)用一次FWD来得到正确的执行顺序
        # FP16 和 FP32不一样
        for name, param in self.module.named_parameters(recurse=True):
            register_param(param, name)
            numel = param.numel()
            data_type = torch.float
            self.chunk_creator.add_tensor(param.ps_attr.data_id(), numel,
                                          param, AccessType.DATA, data_type)
            self.chunk_creator.add_tensor(param.ps_attr.grad_id(), numel,
                                          param, AccessType.DATA, data_type)

        self.chunk_creator.start_new_chunk_list()

        # layout for M, V
        # Parameters`tate[param_group][param]`尚未被初始化，它们在第一次step时候才出现
        # 这里提前计算它们的layout，因此需要将它们提前弄出来
        # TODO(jiaruifang) 不是FWD计算顺序，是init初始化顺序
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # TODO(jiaruifang)应该执行一次backward计算，然后才能知道p的grad是否为None，不能用requires_grad来代替？
                if p.requires_grad is True:

                    state = self.optimizer.state[p]
                    # Eager state initialization, different from Pytorch
                    if len(state) == 0:
                        state['step'] = 0
                        # 被PatrickStar管理
                        # Exponential moving average of gradient values
                        data_type = p.dtype
                        state['exp_avg'] = torch.nn.Parameter(
                            torch.zeros(
                                p.ps_attr.ps_shape,
                                dtype=data_type,
                                # memory_format=torch.preserve_format,
                                device=torch.device('cpu:0')),
                            requires_grad=False)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.nn.Parameter(
                            torch.zeros(
                                p.ps_attr.ps_shape,
                                dtype=data_type,
                                # memory_format=torch.preserve_format,
                                device=torch.device('cpu:0')),
                            requires_grad=False)

                        ps_name_prefix = p.ps_attr.ps_name
                        register_param(state['exp_avg'],
                                       f'{ps_name_prefix}.exp_avg')
                        register_param(state['exp_avg_sq'],
                                       f'{ps_name_prefix}.exp_avg_sq')

                        numel = p.ps_attr.ps_numel
                        self.chunk_creator.add_tensor(
                            state['exp_avg'].ps_attr.data_id(), numel,
                            state['exp_avg'], AccessType.DATA, data_type)

                        self.chunk_creator.add_tensor(
                            state['exp_avg_sq'].ps_attr.data_id(), numel,
                            state['exp_avg_sq'], AccessType.DATA, data_type)

                        # param.data不被需要，将他们的内存释放
                        state['exp_avg'].data = torch.zeros(
                            1, dtype=data_type, device=torch.device('cpu:0'))
                        state['exp_avg_sq'].data = torch.zeros(
                            1, dtype=data_type, device=torch.device('cpu:0'))

                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            raise NotImplementedError
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)
                else:
                    raise RuntimeError
        self.chunk_creator.start_new_chunk_list()

    def schedule_fp16(self):
        """
        为module和optimizer的参数指定chunk schema
        schedule过程为所有parameter注册成ps_tensor
        """

        for name, param in self.module.named_parameters(recurse=True):
            register_param(param, f"{name}")

        # 注册param data fp16，按照初始化顺序
        for group in self.optimizer.param_groups:
            for param in group['params']:
                # register_param(param, "param_fp16")
                numel = param.ps_attr.ps_numel
                data_type = torch.half

                self.chunk_creator.add_tensor(param.ps_attr.data_id(), numel,
                                              param, AccessType.DATA,
                                              data_type)

        self.chunk_creator.start_new_chunk_list()

        # 注册param data fp32
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                data_type = torch.float
                param_fp32 = torch.nn.Parameter(torch.zeros(
                    p.ps_attr.ps_shape,
                    dtype=data_type,
                    device=torch.device('cpu:0')),
                                                requires_grad=False)
                state['fp32_param_data'] = param_fp32
                register_param(param_fp32, f'{p.ps_attr.ps_name}_fp32')
                numel = param_fp32.ps_attr.ps_numel
                self.chunk_creator.add_tensor(param_fp32.ps_attr.data_id(),
                                              numel, param_fp32,
                                              AccessType.DATA, data_type)

        self.chunk_creator.start_new_chunk_list()

        # 注册M，V, param data fp32
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad is True:
                    state = self.optimizer.state[p]
                    state['step'] = 0
                    # 被PatrickStar管理
                    # Exponential moving average of gradient values
                    data_type = torch.float
                    state['exp_avg'] = torch.nn.Parameter(
                        torch.zeros(
                            p.ps_attr.ps_shape,
                            dtype=data_type,
                            # memory_format=torch.preserve_format,
                            device=torch.device('cpu:0')),
                        requires_grad=False)

                    ps_name_prefix = p.ps_attr.ps_name
                    register_param(state['exp_avg'],
                                   f'{ps_name_prefix}.exp_avg')

                    numel = p.ps_attr.ps_numel

                    self.chunk_creator.add_tensor(
                        state['exp_avg'].ps_attr.data_id(), numel,
                        state['exp_avg'], AccessType.DATA, data_type)

                    # param.data不被需要，将他们的内存释放
                    state['exp_avg'].data = torch.zeros(
                        1, dtype=data_type, device=torch.device('cpu:0'))

                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        raise NotImplementedError
                else:
                    raise RuntimeError

        self.chunk_creator.start_new_chunk_list()

        # 注册exp_avg_sq
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.requires_grad is True:
                    state = self.optimizer.state[p]
                    # Eager state initialization, different from Pytorch
                    state['step'] = 0
                    # 被PatrickStar管理
                    # Exponential moving average of gradient values
                    data_type = torch.float
                    state['exp_avg_sq'] = torch.nn.Parameter(
                        torch.zeros(
                            p.ps_attr.ps_shape,
                            dtype=data_type,
                            # memory_format=torch.preserve_format,
                            device=torch.device('cpu:0')),
                        requires_grad=False)

                    ps_name_prefix = p.ps_attr.ps_name
                    register_param(state['exp_avg_sq'],
                                   f'{ps_name_prefix}.exp_avg_sq')

                    numel = p.ps_attr.ps_numel

                    self.chunk_creator.add_tensor(
                        state['exp_avg_sq'].ps_attr.data_id(), numel,
                        state['exp_avg_sq'], AccessType.DATA, data_type)

                    state['exp_avg_sq'].data = torch.zeros(
                        1, dtype=data_type, device=torch.device('cpu:0'))

                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        raise NotImplementedError
                else:
                    raise RuntimeError

        # 收尾
        self.chunk_creator.start_new_chunk_list()

        logging.info('static schedule with FP16 finished')
