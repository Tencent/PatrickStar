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


def F_adam(
        client,
        params: List[torch.nn.Parameter],
        #  grads: List[Tensor],
        exp_avgs: List[torch.nn.Parameter],
        exp_avg_sqs: List[torch.nn.Parameter],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """
    logging.debug('begin compute F_adam')
    for i, param in enumerate(params):
        # HybridPS加载data
        compute_device = torch.device('cpu')
        client.access_data(param, compute_device)
        client.access_grad(param, compute_device)

        param_data = param.data
        param_grad = param.grad

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]

        client.access_data(exp_avg, compute_device)
        client.access_data(exp_avg_sq, compute_device)

        exp_avg_data = exp_avg.data
        exp_avg_sq_data = exp_avg_sq.data

        step = state_steps[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        if weight_decay != 0:
            param_grad = param_grad.add(param_data, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg_data.mul_(beta1).add_(param_grad, alpha=1 - beta1)
        exp_avg_sq_data.mul_(beta2).addcmul_(param_grad,
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
            denom = (exp_avg_sq_data.sqrt() /
                     math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg_data, denom, value=-step_size)

        client.release_data(param)
        client.release_grad(param)
        client.release_data(exp_avg)
        client.release_data(exp_avg_sq)

    logging.debug('finish compute F_adam')


class CPUAdam(torch.optim.Optimizer):
    def __init__(self,
                 client,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False):
        """
        父类Optimzer实现细节
        https://github.com/pytorch/pytorch/blob/c371542efc/torch/optim/optimizer.py
        需要在register_module之前调用？也许不用，只用param的地址
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

    def __setstate__(self, state):
        super(CPUAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def warmup(self):
        """
      初始化M (exp_avg)，V (exp_avg_sq)的空间，在训练前运行
      """
        for group_id, group in enumerate(self.param_groups):
            for param_id, p in enumerate(group['params']):

                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    #print(f'group {group_id} param {param_id} = {p.numel()}')
                    state['step'] = 0
                    # gradient momentums
                    state['exp_avg'] = torch.zeros_like(p.data,
                                                        dtype=p.dtype,
                                                        device='cpu')
                    #memory_format=torch.preserve_format)
                    # gradient variances
                    state['exp_avg_sq'] = torch.zeros_like(p.data,
                                                           dtype=p.dtype,
                                                           device='cpu')

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'Adam does not support sparse gradients, please consider SparseAdam instead'
                        )
                    # grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # 被HybridPS管理
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.nn.Parameter(
                            torch.zeros_like(
                                p,
                                memory_format=torch.preserve_format,
                                device=torch.device('cpu')),
                            requires_grad=False)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.nn.Parameter(
                            torch.zeros_like(
                                p,
                                memory_format=torch.preserve_format,
                                device=torch.device('cpu')),
                            requires_grad=False)

                        assert state['exp_avg'].requires_grad is False
                        assert state['exp_avg_sq'].requires_grad is False
                        self.client.register_param(state['exp_avg'])
                        self.client.register_param(state['exp_avg_sq'])

                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            F_adam(
                self.client,
                params_with_grad,
                #    grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                group['amsgrad'],
                beta1,
                beta2,
                group['lr'],
                group['weight_decay'],
                group['eps'])
        return loss
