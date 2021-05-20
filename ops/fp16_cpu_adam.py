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

import math
import torch
import time
from pathlib import Path
from torch import Tensor
from typing import List, Optional
import logging
from client.const import PSTensorStatus, AccessType
import utils.global_timer as global_timer
from client.parameter import register_param


def FP16_f_adam(client,
                fp32_params: List[torch.nn.Parameter],
                fp16_param_with_grad,
                exp_avgs: List[torch.nn.Parameter],
                exp_avg_sqs: List[torch.nn.Parameter],
                max_exp_avg_sqs: List[Tensor],
                state_steps: List[int],
                amsgrad: bool,
                beta1: float,
                beta2: float,
                lr: float,
                weight_decay: float,
                eps: float,
                max_param_size,
                param_grad_buff,
                prefer_device,
                time_profile=True):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """
    timer = global_timer.IterationTimer()
    if time_profile:
        adam_start_time = time.time()
    # TODO(jiaruifang)计算粒度为什么是tensor，而不是chunk
    for i, param in enumerate(fp32_params):
        if time_profile:
            adam_iter_access_start = time.time()

        compute_device = prefer_device
        client.access_data(param, compute_device)
        param_data = param.ps_attr.access_tensor(AccessType.DATA)

        param_grad = param_grad_buff.narrow(0, 0, param_data.numel()).view(
            param_data.shape)
        fp16_param = fp16_param_with_grad[i]
        client.access_grad(fp16_param, torch.device('cuda:0'))
        fp16_param_grad = fp16_param.ps_attr.access_tensor(AccessType.GRAD)

        if time_profile:
            start_time = time.time()
        # torch.cuda.synchronize()
        param_grad.copy_(fp16_param_grad, non_blocking=False)
        # torch.cuda.synchronize()
        if time_profile:
            global_timer.cpu_gpu_move_elapse += time.time() - start_time
            global_timer.cpu_gpu_move_times += 1
            global_timer.cpu_gpu_move_data_amount += param_grad.numel()

        #TODO(jiaruifang) HOLD->FREE
        fp16_param_grad.zero_()
        client.release_grad(fp16_param, PSTensorStatus.FREE)

        exp_avg_param = exp_avgs[i]
        exp_avg_sq_param = exp_avg_sqs[i]

        client.access_data(exp_avg_param, compute_device)
        client.access_data(exp_avg_sq_param, compute_device)

        exp_avg = exp_avg_param.ps_attr.access_tensor(AccessType.DATA)

        exp_avg_sq = exp_avg_sq_param.ps_attr.access_tensor(AccessType.DATA)

        if time_profile:
            global_timer.cpu_adam_access_elapse += time.time(
            ) - adam_iter_access_start
            f_adam_compute_start_time = time.time()

        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            param_grad = param_grad.add(param_data, alpha=weight_decay)

        exp_avg.mul_(beta1).add_(param_grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(param_grad,
                                        param_grad,
                                        value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i],
                          exp_avg_sq,
                          out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() /
                     math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param_data.addcdiv_(exp_avg, denom, value=-step_size)

        if time_profile:
            global_timer.cpu_adam_f_elapse += time.time(
            ) - f_adam_compute_start_time
            adam_iter_release_start = time.time()

        fp16_param = fp16_param_with_grad[i]
        client.access_data(fp16_param, torch.device('cuda:0'))
        fp16_data = fp16_param.ps_attr.access_tensor(AccessType.DATA)
        if time_profile:
            start_time = time.time()
        fp16_data.copy_(param_data, non_blocking=False)
        if time_profile:
            global_timer.gpu_cpu_move_elapse += time.time() - start_time
            global_timer.gpu_cpu_move_data_amount += fp16_data.numel()
            global_timer.gpu_cpu_move_times += 1
        client.release_data(fp16_param, PSTensorStatus.HOLD)

        client.release_data(param)
        client.release_data(exp_avg_param)
        client.release_data(exp_avg_sq_param)

        if time_profile:
            global_timer.cpu_adam_release_elapse += time.time(
            ) - adam_iter_release_start

    timer.tik(device_type='all')
    global_timer.cpu_adam_elapse += time.time() - adam_start_time


class FP16Adam(torch.optim.Optimizer):
    def __init__(self,
                 client,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 prefer_device=torch.device('cpu:0')):
        """
        父类Optimzer实现细节
        https://github.com/pytorch/pytorch/blob/c371542efc/torch/optim/optimizer.py
        需要在register_module之前调用？也许不用，只用param的地址
        TODO(jiaruifang) prefer_device应该是自适应的
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(
                betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(
                betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        amsgrad=amsgrad)
        super(FP16Adam, self).__init__(params, defaults)
        self.client = client
        self.prefer_device = prefer_device

        max_param_size = 0
        data_type = None
        for group in self.param_groups:
            for p in group['params']:
                max_param_size = max(max_param_size, p.numel())
                data_type = p.dtype

        self.max_param_size = max_param_size
        assert data_type == torch.half
        if self.prefer_device.type == 'cpu':
            self.param_grad_buff = torch.zeros(max_param_size,
                                               dtype=torch.float,
                                               device=self.prefer_device,
                                               pin_memory=True)
        else:
            self.param_grad_buff = torch.zeros(max_param_size,
                                               dtype=torch.float,
                                               device=self.prefer_device)

        # 存储fp32 param的data
        # 在初始化先把fp16的数据拷贝到fp32的参数内
        # 按照初始化顺序来拷贝
        self.fp32_params_list = []
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                state = self.state[p]
                state['step'] = 0

                fp32_param = torch.nn.Parameter(torch.zeros_like(
                    p, dtype=torch.float, device=torch.device('cpu:0')),
                                                requires_grad=False)
                self.client.access_data(fp32_param, self.prefer_device)
                fp32_param_data = fp32_param.ps_attr.access_tensor(
                    AccessType.DATA)
                fp32_param_data.copy_(p.data.float())
                state['fp32_param_data'] = fp32_param
                self.client.release_data(fp32_param)

                state['exp_avg'] = torch.nn.Parameter(torch.zeros(
                    p.shape, dtype=torch.float, device=self.prefer_device),
                                                      requires_grad=False)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'] = torch.nn.Parameter(torch.zeros(
                    p.shape, dtype=torch.float, device=self.prefer_device),
                                                         requires_grad=False)

                self.client.access_data(state['exp_avg'], self.prefer_device)
                state['exp_avg'].ps_attr.access_tensor(AccessType.DATA).zero_()
                self.client.release_data(state['exp_avg'])

                self.client.access_data(state['exp_avg_sq'],
                                        self.prefer_device)
                state['exp_avg_sq'].ps_attr.access_tensor(
                    AccessType.DATA).zero_()
                self.client.release_data(state['exp_avg_sq'])

    def __setstate__(self, state):
        super(CPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # 对hook逻辑bug进行补救，embedding层的grad反向结束后仍是COMPUTE，这里将它们设置为HOLD
        # fp16 逻辑
        for n, p in self.client.module.named_parameters():
            self.client.release_grad(p, PSTensorStatus.HOLD)
            self.client.release_data(p, PSTensorStatus.FREE)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            fp16_param_with_grad = []
            fp32_param = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for j, p in enumerate(group['params']):
                if p.requires_grad:
                    fp16_param_with_grad.append(p)
                    state = self.state[p]

                    # 初始化M，V，FP32 data
                    # 把原来的FP32 data释放掉
                    if len(state) == 0:
                        # 第一次预热时候，拷贝FP32 data数据
                        state['step'] = 0
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    fp32_param.append(state['fp32_param_data'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
                else:
                    raise RuntimeError(f"tensor id {p.ps_attr.grad_id()}")

            beta1, beta2 = group['betas']

            # self.client.chunk_tensor_index.visit_chunks(self.client.chunk_list)
            # input('wait')
            FP16_f_adam(
                self.client, fp32_param, fp16_param_with_grad, exp_avgs,
                exp_avg_sqs, max_exp_avg_sqs, state_steps, group['amsgrad'],
                beta1, beta2, group['lr'], group['weight_decay'], group['eps'],
                self.max_param_size, self.param_grad_buff if hasattr(
                    self, 'param_grad_buff') else None, self.prefer_device)

        return loss
