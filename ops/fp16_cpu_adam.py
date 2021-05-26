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


def FP16_f_adamv2(client,
                  fp32_params: List[torch.nn.Parameter],
                  fp16_param_with_grad_list,
                  exp_avgs: List[torch.nn.Parameter],
                  exp_avg_sqs: List[torch.nn.Parameter],
                  max_exp_avg_sqs: List[Tensor],
                  state_steps: List[int],
                  amsgrad: bool,
                  beta1_list: List[float],
                  beta2_list: List[float],
                  lr_list: List[float],
                  weight_decay_list: List[float],
                  eps_list: List[float],
                  prefer_device,
                  param_grad_buff,
                  time_profile=True):
    r"""Functional API that performs Adam algorithm computation.
    按照在chunk内的存储顺序连续访问fp16_param_with_grad_list的参数，获取fp16 grad，
    以chunk为单位拷贝到一个tmp buff之中
    """
    # assert prefer_device.type == 'cpu'
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

        fp16_param = fp16_param_with_grad_list[i]

        # 把fp16_param所在的chunk拷贝到tmp_buff中，并返回对应的tensor
        if False:
            # client.access_grad(fp16_param, torch.device(f'cuda:{client.rank}'))
            param_grad = client.fp16_to_fp32_copy(
                fp16_param, AccessType.DATA).view(param_data.shape)
            # necessary to reset grads
            client.release_data(fp16_param, PSTensorStatus.FREE)
        else:
            # 放在data位置上的grad
            client.access_data(fp16_param, torch.device(f'cuda:{client.rank}'))
            fp16_param_grad = fp16_param.ps_attr.access_tensor(AccessType.DATA)
            if time_profile:
                start_time = time.time()
            param_grad = param_grad_buff.narrow(0, 0, param_data.numel()).view(
                param_data.shape)
            # torch.cuda.synchronize()
            # print(f"fp16 ps grad {i} ", fp16_param_grad)
            param_grad.copy_(fp16_param_grad, non_blocking=False)
            # torch.cuda.synchronize()
            if time_profile:
                global_timer.gpu_cpu_move_elapse += time.time() - start_time
                global_timer.gpu_cpu_move_times += 1
                global_timer.gpu_cpu_move_data_amount += param_grad.numel()

            client.release_data(fp16_param, PSTensorStatus.FREE)

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
        beta1 = beta1_list[i]
        beta2 = beta2_list[i]
        eps = eps_list[i]

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        weight_decay = weight_decay_list[i]

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

        lr = lr_list[i]
        step_size = lr / bias_correction1

        param_data.addcdiv_(exp_avg, denom, value=-step_size)

        if time_profile:
            global_timer.cpu_adam_f_elapse += time.time(
            ) - f_adam_compute_start_time
            adam_iter_release_start = time.time()

        fp16_param = fp16_param_with_grad_list[i]

        client.access_data(fp16_param, torch.device(f'cuda:{client.rank}'))

        fp16_data = fp16_param.ps_attr.access_tensor(AccessType.DATA)
        if time_profile:
            start_time = time.time()
        # TODO 直接拷贝一块
        fp16_data.copy_(param_data, non_blocking=False)
        if time_profile:
            global_timer.cpu_gpu_move_elapse += time.time() - start_time
            global_timer.cpu_gpu_move_data_amount += fp16_data.numel()
            global_timer.cpu_gpu_move_times += 1

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

        # 将group参数放置到每个param内部
        for group in self.param_groups:
            for p in group['params']:
                if p.dtype == torch.float:
                    p.data = p.data.half()
                self.state[p]['betas'] = group['betas']
                self.state[p]['lr'] = group['lr']
                self.state[p]['weight_decay'] = group['weight_decay']
                self.state[p]['eps'] = group['eps']

        self.param_grad_buff = None

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
        # 对hook逻辑bug进行补救，第一层层的grad反向结束后仍是COMPUTE，这里将它们设置为HOLD
        for n, param in self.client.module.named_parameters():
            if param.ps_attr.get_status(
                    AccessType.DATA) != PSTensorStatus.HOLD:
                tmp_tensor = param.ps_attr.access_tensor(AccessType.DATA)
                tmp_tensor.copy_(param.grad)
                param.grad = None
                self.client.release_data(param, PSTensorStatus.HOLD)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        fp16_param_with_grad_list = []
        fp32_param_list = []
        exp_avgs = []
        exp_avg_sqs = []
        state_sums = []
        max_exp_avg_sqs = []
        state_steps = []
        beta1_list = []
        beta2_list = []
        weight_decay_list = []
        eps_list = []
        lr_list = []

        self.client._cached_fp32_buff.reset()
        logging.info('init adam')
        first_init_flag = False

        # for p in self.client.generate_grad_params():
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if p.requires_grad:
                    fp16_param_with_grad_list.append(p)
                    state = self.state[p]

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    fp32_param_list.append(state['fp32_param_data'])
                    beta1, beta2 = state['betas']

                    beta1_list.append(beta1)
                    beta2_list.append(beta2)
                    lr_list.append(state['lr'])
                    weight_decay_list.append(state['weight_decay'])
                    eps_list.append(state['eps'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])
                else:
                    raise RuntimeError(f"tensor id {p.ps_attr.grad_id()}")

        if self.param_grad_buff is None:
            if self.prefer_device.type == 'cpu':
                self.param_grad_buff = torch.zeros(
                    self.client.chunk_list.max_chunk_size(),
                    dtype=torch.float,
                    device=self.prefer_device,
                    pin_memory=True)
            else:
                self.param_grad_buff = torch.zeros(
                    self.client.chunk_list.max_chunk_size(),
                    dtype=torch.float,
                    device=self.prefer_device)
            logging.info(
                f"adam max_chunk_size {self.client.chunk_list.max_chunk_size()}"
            )
        FP16_f_adamv2(self.client, fp32_param_list, fp16_param_with_grad_list,
                      exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps,
                      False, beta1_list, beta2_list, lr_list,
                      weight_decay_list, eps_list, self.prefer_device,
                      self.param_grad_buff)
        return loss
