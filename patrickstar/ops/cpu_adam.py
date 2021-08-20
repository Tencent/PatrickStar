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
from patrickstar.core.const import PSTensorStatus
import patrickstar.utils.global_timer as global_timer


def F_adam(client, params: List[torch.nn.Parameter],
           exp_avgs: List[torch.nn.Parameter],
           exp_avg_sqs: List[torch.nn.Parameter],
           max_exp_avg_sqs: List[Tensor], state_steps: List[int],
           amsgrad: bool, beta1: float, beta2: float, lr: float,
           weight_decay: float, eps: float, prefer_device):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """
    adam_start_time = time.time()
    # TODO(jiaruifang)计算粒度为什么是tensor，而不是chunk
    for i, param in enumerate(params):
        adam_iter_access_start = time.time()
        compute_device = prefer_device
        client.access_data(param, compute_device)
        client.access_grad(param, compute_device)
        param_data = param.ps_attr.access_tensor(AccessType.DATA)
        param_grad = param.ps_attr.access_tensor(AccessType.GRAD)

        exp_avg_param = exp_avgs[i]
        exp_avg_sq_param = exp_avg_sqs[i]

        client.access_data(exp_avg_param, compute_device)
        client.access_data(exp_avg_sq_param, compute_device)

        exp_avg = exp_avg_param.ps_attr.access_tensor(AccessType.DATA)
        exp_avg_sq = exp_avg_sq_param.ps_attr.access_tensor(AccessType.DATA)

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

        global_timer.cpu_adam_f_elapse += time.time(
        ) - f_adam_compute_start_time

        adam_iter_release_start = time.time()
        # TODO(jiarufiang) release grad to FREE or to HOLD?
        # to HOLD to avoild memory release. but occupying more memory. (reset to zero to make sure correct answer)
        # to FREE will release memory (if not optimized).
        param_grad.zero_()
        client.release_grad(param, PSTensorStatus.HOLD)

        client.release(param)
        client.release(exp_avg_param)
        client.release(exp_avg_sq_param)
        global_timer.cpu_adam_release_elapse += time.time(
        ) - adam_iter_release_start

    global_timer.cpu_adam_elapse += time.time() - adam_start_time


class CPUAdam(torch.optim.Optimizer):
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
        super(CPUAdam, self).__init__(params, defaults)
        self.client = client
        self.prefer_device = prefer_device

        # fp16才需要
        max_param_size = 0
        data_type = None
        for group in self.param_groups:
            for p in group['params']:
                max_param_size = max(max_param_size, p.numel())
                data_type = p.dtype

        self.max_param_size = max_param_size
        if data_type == torch.half:
            if self.prefer_device.type == 'cpu':
                self.param_grad_buff = torch.zeros(max_param_size,
                                                   dtype=torch.float,
                                                   device=self.prefer_device,
                                                   pin_memory=True)
            else:
                self.param_grad_buff = torch.zeros(max_param_size,
                                                   dtype=torch.float,
                                                   device=self.prefer_device)

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
        for n, p in self.client.module.named_parameters():
            self.client.release_grad(p, PSTensorStatus.HOLD)
            self.client.release(p, PSTensorStatus.HOLD)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for i, group in enumerate(self.param_groups):
            params_with_grad = []
            # grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for j, p in enumerate(group['params']):
                if p.requires_grad:
                    params_with_grad.append(p)

                    state = self.state[p]
                    # 以下逻辑在ChunkSchemaScheduler中
                    assert len(state) != 0

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
                else:
                    raise RuntimeError(f"tensor id {p.ps_attr.grad_id()}")

            beta1, beta2 = group['betas']
            F_adam(self.client, params_with_grad, exp_avgs, exp_avg_sqs,
                   max_exp_avg_sqs, state_steps, group['amsgrad'], beta1,
                   beta2, group['lr'], group['weight_decay'], group['eps'],
                   self.prefer_device)

        return loss
