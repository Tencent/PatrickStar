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
from .parameter import PSParameter
import torch
import logging


class ChunkShemaScheduler(object):
    def __init__(self, default_chunk_size, module, optimizer, chunk_list,
                 chunk_tensor_index: ChunkTensorIndex):
        self.module = module
        self.optimizer = optimizer
        self.default_chunk_size = default_chunk_size
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index

    def register_param(self, param, name=None):
        param.ps_attr = PSParameter(param, name)

    def schedule(self):
        """
        为module和optimizer的参数指定chunk schema
        schedule过程为所有parameter进行ps化
        """
        # 模型参数的data和grad间隔排列。
        # Optimizer的M，V间隔排列
        acc_cnt = 0
        chunk_id = 0

        # 注册model和optimizer的param是否应该和chunk layout schedule耦合？
        # FP16和FP32都需要注册的
        for name, param in self.module.named_parameters(recurse=True):
            self.register_param(param, name)
            numel = param.numel()
            data_type = param.dtype
            self.chunk_tensor_index.add_tensor(chunk_id,
                                               param.ps_attr.data_id(),
                                               acc_cnt, numel, param,
                                               AccessType.DATA)
            self.chunk_tensor_index.add_tensor(chunk_id,
                                               param.ps_attr.grad_id(),
                                               acc_cnt + numel, numel, param,
                                               AccessType.GRAD)
            acc_cnt += numel * 2
            if acc_cnt >= self.default_chunk_size:
                self.chunk_list.new_chunk(chunk_id, acc_cnt, data_type,
                                          torch.device('cuda:0'))
                chunk_id += 1
                acc_cnt = 0

        # 收尾，剩下的tensor凑不成一个至少default size大小的chunk
        if acc_cnt > 0:
            self.chunk_list.new_chunk(chunk_id, acc_cnt, data_type,
                                      torch.device('cuda:0'))
            chunk_id += 1
            acc_cnt = 0

        if hasattr(self.optimizer, "fp32_from_fp16_groups"):
            logging.info(f'schedule for fp16 fp32_from_fp16_groups')
            for param_group in self.optimizer.fp32_from_fp16_groups:
                for param in param_group:
                    # TODO, 还不能获取name
                    self.register_param(param, 'master')
                    numel = param.ps_attr.ps_numel
                    data_type = param.dtype
                    self.chunk_tensor_index.add_tensor(chunk_id,
                                                       param.ps_attr.data_id(),
                                                       acc_cnt, numel, param,
                                                       AccessType.DATA)
                    self.chunk_tensor_index.add_tensor(chunk_id,
                                                       param.ps_attr.grad_id(),
                                                       acc_cnt + numel, numel,
                                                       param, AccessType.GRAD)
                    acc_cnt += numel * 2
                    if acc_cnt > self.default_chunk_size:
                        self.chunk_list.new_chunk(chunk_id, acc_cnt, data_type,
                                                  torch.device('cuda:0'))
                        chunk_id += 1
                        acc_cnt = 0
            # 收尾
            if acc_cnt > 0:
                self.chunk_list.new_chunk(chunk_id, acc_cnt, data_type,
                                          torch.device('cpu:0'))
                chunk_id += 1
                acc_cnt = 0

        # layout for M, V
        # Parameters`tate[param_group][param]`尚未被初始化，它们在第一次step时候才出现
        # 这里提前计算它们的layout，因此需要将它们提前弄出来
        for group in self.optimizer.param_groups:
            for p in group['params']:
                # TODO(jiaruifang)应该执行一次backward计算，然后才能知道p的grad是否为None，不能用requires_grad来代替？
                if p.requires_grad is True:
                    # params_with_grad.append(p)
                    # if p.ps_grad_tensor.is_sparse:
                    #     raise RuntimeError(
                    #         'Adam does not support sparse gradients, please consider SparseAdam instead'
                    #     )
                    # grads.append(p.grad)

                    state = self.optimizer.state[p]
                    # Eager state initialization, different from Pytorch
                    if len(state) == 0:
                        state['step'] = 0
                        # 被HybridPS管理
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
                        self.register_param(state['exp_avg'],
                                            f'{ps_name_prefix}.exp_avg')
                        self.register_param(state['exp_avg_sq'],
                                            f'{ps_name_prefix}.exp_avg_sq')

                        numel = p.ps_attr.ps_numel
                        self.chunk_tensor_index.add_tensor(
                            chunk_id, state['exp_avg'].ps_attr.data_id(),
                            acc_cnt, numel, state['exp_avg'], AccessType.DATA)
                        self.chunk_tensor_index.add_tensor(
                            chunk_id, state['exp_avg_sq'].ps_attr.data_id(),
                            acc_cnt + numel, numel, state['exp_avg_sq'],
                            AccessType.DATA)

                        # param.data不被需要，将他们的内存释放
                        state['exp_avg'].data = torch.zeros(
                            1, dtype=data_type, device=torch.device('cpu:0'))
                        state['exp_avg_sq'].data = torch.zeros(
                            1, dtype=data_type, device=torch.device('cpu:0'))

                        acc_cnt += numel * 2
                        # logging.info(f'acc_cnt {acc_cnt}')
                        if acc_cnt >= self.default_chunk_size:
                            # logging.info(f'here')
                            self.chunk_list.new_chunk(chunk_id, acc_cnt,
                                                      torch.float,
                                                      torch.device('cpu:0'))
                            chunk_id += 1
                            acc_cnt = 0

                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            raise NotImplementedError
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)
                else:
                    raise RuntimeError

        # 收尾
        if acc_cnt > 0:
            self.chunk_list.new_chunk(chunk_id, acc_cnt, data_type,
                                      torch.device('cpu:0'))
