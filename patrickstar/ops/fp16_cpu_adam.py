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

from patrickstar.core.const import PSTensorStatus, AccessType, TrainingStage
import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import print_rank, logger, use_dist_flag, get_sys_memory_used
from patrickstar.core.parameter import register_param, is_torch_param, register_torch_param
from patrickstar.deepspeed_helper.global_vars import get_args
from patrickstar.manager import PatrickStarManager
from patrickstar.core import ChunkList, ChunkTensorIndex
from .chunk_io_buff import FP32ChunkReadBuffer, FP16ChunkWriteBuffer

from .op_builder import CPUAdamBuilder


def get_real_data_tensor(param):
    if is_torch_param(param):
        return param
    else:
        return param.ps_attr.access_tensor(AccessType.DATA)


class FP16Adam(torch.optim.Optimizer):
    optimizer_id = 0

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

        # Eager state initialization, different from Pytorch
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                # 将group参数放置到每个param内部，可以按照参数切分并行计算adam
                self.state[p]['betas'] = group['betas']
                self.state[p]['lr'] = group['lr']
                self.state[p]['weight_decay'] = group['weight_decay']
                self.state[p]['eps'] = group['eps']

                state['step'] = 0

                if is_torch_param(p):
                    state['fp32_param_data'] = torch.nn.Parameter(
                        torch.zeros_like(p, dtype=torch.float, device=torch.device('cpu:0')),
                                         requires_grad=False)
                    register_torch_param(state['fp32_param_data'])

                    if p.requires_grad:
                        state['exp_avg'] = torch.nn.Parameter(
                            torch.zeros_like(p, dtype=torch.float, device=torch.device('cpu:0')),
                                            requires_grad=False)
                        register_torch_param(state['exp_avg'])
                        state['exp_avg_sq'] = torch.nn.Parameter(
                            torch.zeros_like(p, dtype=torch.float, device=torch.device('cpu:0')),
                                            requires_grad=False)
                        register_torch_param(state['exp_avg_sq'])
                else:
                    state['fp32_param_data'] = torch.nn.Parameter(
                        torch.tensor([], dtype=torch.float, device=torch.device('cpu:0')),
                                     requires_grad=False)
                    register_param(state['fp32_param_data'], f'{p.ps_attr.name}_fp32')
                    state['fp32_param_data'].ps_attr.reset_shape(p.ps_attr.shape)

                    if p.requires_grad:
                        state['exp_avg'] = torch.nn.Parameter(
                            torch.tensor([], dtype=torch.float,
                                        # memory_format=torch.preserve_format,
                                        device=torch.device('cpu:0')),
                                        requires_grad=False)
                        register_param(state['exp_avg'], f'{p.ps_attr.name}.exp_avg')
                        state['exp_avg'].ps_attr.reset_shape(p.ps_attr.shape)

                        state['exp_avg_sq'] = torch.nn.Parameter(
                            torch.tensor([], dtype=torch.float,
                                        # memory_format=torch.preserve_format,
                                        device=torch.device('cpu:0')),
                            requires_grad=False)
                        register_param(state['exp_avg_sq'], f'{p.ps_attr.name}.exp_avg_sq')
                        state['exp_avg_sq'].ps_attr.reset_shape(p.ps_attr.shape)


        # 用作fp16 grad 存储的buffer
        self.read_chunk_buff = None
        args = get_args()
        self.use_ds_adam = args.use_deepspeed_cpu_adam
        if self.use_ds_adam:
            self.opt_id = FP16Adam.optimizer_id
            FP16Adam.optimizer_id = FP16Adam.optimizer_id + 1
            self.ds_opt_adam = CPUAdamBuilder().load()
            self.ds_opt_adam.create_adam(self.opt_id, lr, betas[0], betas[1],
                                         eps, weight_decay, False, True)

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when deepspeed.initialize
        # is used multiple times in the same process (notebook or pytest worker)
        if self.use_ds_adam:
            self.ds_opt_adam.destroy_adam(self.opt_id)

    def __setstate__(self, state):
        super(FP16Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def FP16_f_adamv2(self,
                      client,
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
                      read_chunk_buff,
                      write_chunk_buff,
                      time_profile=True,
                      margin_chunk_num_for_gpu_adam=0):
        r"""Functional API that performs Adam algorithm computation.
        按照在chunk内的存储顺序连续访问fp16_param_with_grad_list的参数，获取fp16 grad，
        以chunk为单位拷贝到一个tmp buff之中
        """
        args = get_args()
        if args.use_fake_dist:
            rank = 0
        else:
            rank = args.local_rank
        world_size = torch.distributed.get_world_size()
        logger.info(
            f'rank {rank} margin_chunk_num_for_gpu_adam {margin_chunk_num_for_gpu_adam}, '
            f'param cnt {len(fp32_params)}'
        )
        for i, fp32_param in enumerate(fp32_params):
            ##########################
            ####### 准备ADAM数据 ######
            ##########################
            fp16_param = fp16_param_with_grad_list[i]

            if time_profile:
                global_timer.my_timer.start_profile('ADAM_prepare_data')
                global_timer.my_timer.start_profile(
                    'ADAM_prepare_data_fp16_grad_to_fp32_grad_copy')

            # 以chunk为粒度拷贝grad fp16 (FWD+BWD计算设备GPU, CPU如果被换出了) -> grad fp32 (Adam计算设备CPU or GPU如果margin空间足够)
            if is_torch_param(fp16_param):
                # 如果fp16_param被Torch管理，则它肯定在cpu上，cpu_embedding优化引起的
                assert fp16_param.data.device.type == 'cpu'
                fp16_grad_tensor = fp16_param.data
            else:
                # 将FP16 GPU Chunk拷贝到compute_device的FP32 Chunk上。
                # 如果是第一个tensor则拷贝Chunk，否则索引chunk
                fp16_grad_tensor = read_chunk_buff.access_from_cache(
                    fp16_param).view(fp16_param.ps_attr.shape)

            compute_device = fp16_grad_tensor.device
            logger.debug(
                f'rank {args.local_rank} adam {i} on {compute_device}')

            if time_profile:
                global_timer.my_timer.finish_profile(
                    'ADAM_prepare_data_fp16_grad_to_fp32_grad_copy')
                global_timer.data_move_cnter.update(
                    'ADAM_prepare_data_fp16_grad_to_fp32_grad_copy',
                    fp16_grad_tensor.numel() * 2)

            client.access_data(fp32_param, compute_device)
            fp32_data_tensor = get_real_data_tensor(fp32_param)

            exp_avg_param = exp_avgs[i]
            exp_avg_sq_param = exp_avg_sqs[i]

            client.access_data(exp_avg_param, compute_device)
            client.access_data(exp_avg_sq_param, compute_device)

            exp_avg = get_real_data_tensor(exp_avg_param)
            exp_avg_sq = get_real_data_tensor(exp_avg_sq_param)

            ##########################
            ####### 开始ADAM计算 ######
            ##########################
            if time_profile:
                global_timer.my_timer.finish_profile('ADAM_prepare_data')
                global_timer.my_timer.start_profile('ADAM_compute')

            step = state_steps[i]
            beta1 = beta1_list[i]
            beta2 = beta2_list[i]
            eps = eps_list[i]

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            weight_decay = weight_decay_list[i]
            lr = lr_list[i]

            # TODO(jiaruifang) use_ds_adam时，在生成的数据上正确性没有验证
            if self.use_ds_adam and compute_device.type == 'cpu' and fp16_grad_tensor.device.type == 'cpu':
                assert fp32_data_tensor.device.type == 'cpu'
                assert fp16_grad_tensor.device.type == 'cpu'
                assert exp_avg.device.type == 'cpu'
                assert exp_avg_sq.device.type == 'cpu'

                # Inputs of DS CPU Adam need to be flattened.
                self.ds_opt_adam.adam_update(self.opt_id, step, lr, beta1,
                                             beta2, eps, weight_decay, True,
                                             fp32_data_tensor.view(-1),
                                             fp16_grad_tensor.view(-1), exp_avg.view(-1),
                                             exp_avg_sq.view(-1))
            else:
                fp32_grad_tensor = fp16_grad_tensor.float()
                if weight_decay != 0:
                    fp32_grad_tensor = fp32_grad_tensor.add(fp32_data_tensor,
                                                            alpha=weight_decay)

                exp_avg.mul_(beta1).add_(fp32_grad_tensor, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(fp32_grad_tensor,
                                                fp32_grad_tensor,
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
                    denom = (exp_avg_sq.sqrt() /
                             math.sqrt(bias_correction2)).add_(eps)

                step_size = lr / bias_correction1

                fp32_data_tensor.addcdiv_(exp_avg, denom, value=-step_size)

            if time_profile:
                global_timer.my_timer.finish_profile('ADAM_compute')
                global_timer.my_timer.start_profile('ADAM_param_fp32_to_fp16')

            ##########################
            ####### 结束ADAM计算 ######
            ##########################

            # Note fp16_param对应的Chunk内存 ->fp32_param对应的chunk内存
            write_chunk_buff.write_from_cache(fp16_param, fp32_param)

            if time_profile:
                global_timer.my_timer.finish_profile('ADAM_param_fp32_to_fp16')
                global_timer.data_move_cnter.update(
                    'ADAM_param_fp32_to_fp16',
                    fp32_data_tensor.numel() * 4)
                global_timer.my_timer.start_profile('ADAM_release_data')

            client.release_data(fp32_param)
            client.release_data(exp_avg_param)
            client.release_data(exp_avg_sq_param)

            if time_profile:
                global_timer.my_timer.finish_profile('ADAM_release_data')

            # 预热时记录内存使用情况
            mgr = PatrickStarManager()
            mgr.tiktac(client)

        write_chunk_buff.reset()
        read_chunk_buff.reset()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        args = get_args()
        rank = torch.distributed.get_rank()
        # Here we need to use module.parameter() order
        # because it is the order of params in the chunk_list
        for n, param in self.client.module.named_parameters():
            if is_torch_param(param) and param.grad is not None:
                param.data = param.grad
                param.grad = None
                world_size = torch.distributed.get_world_size()

                torch.distributed.all_reduce(param.data,
                                             op=torch.distributed.ReduceOp.SUM,
                                             group=self.client.cpu_comm_group,
                                             async_op=False)
                param.data /= world_size

                logger.info(
                    f'rank {rank} allreduce grad {param.ps_attr.name}')
                continue
            if param.ps_attr.get_status(
                    AccessType.DATA) == PSTensorStatus.COMPUTE:
                logger.debug(
                    f'rank {rank} release param {n} from COMPUTE to HOLD_AFTER_BWD'
                )
                tmp_tensor = param.ps_attr.access_tensor(AccessType.DATA)
                tmp_tensor.copy_(param.grad)
                param.grad = None

                if torch.distributed.is_initialized():
                    if use_dist_flag:
                        self.client.release_dist(
                            param,
                            AccessType.DATA,
                            PSTensorStatus.HOLD_AFTER_BWD,
                            training_stage=TrainingStage.BWD,
                            is_allreduce=True)
                    else:
                        self.client.release(param, AccessType.DATA,
                                            PSTensorStatus.HOLD_AFTER_BWD,
                                            True)
                else:
                    self.client.release_data(param, PSTensorStatus.HOLD)
        mgr = PatrickStarManager()
        mgr._training_stage = TrainingStage.ADAM
        logger.info(f'Entering ADAM Stage')

        mgr.tiktac(self.client)

        global_timer.my_timer.start_profile('ADAM')
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

        max_param_size = 0
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if p.requires_grad:
                    # update the steps for each param group update
                    state = self.state[p]
                    state['step'] += 1

                    # p不是torch param，且p属于remote chunk跳过
                    if use_dist_flag and not is_torch_param(
                            p) and not self.client.is_local_tensor(
                                p, AccessType.DATA):
                        continue

                    if is_torch_param(p):
                        max_param_size = max(p.numel(), max_param_size)

                    fp16_param_with_grad_list.append(p)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    fp32_param_list.append(state['fp32_param_data'])
                    beta1, beta2 = state['betas']

                    beta1_list.append(beta1)
                    beta2_list.append(beta2)
                    lr_list.append(state['lr'])
                    weight_decay_list.append(state['weight_decay'])
                    eps_list.append(state['eps'])

                    # record the step after step update
                    state_steps.append(state['step'])
                else:
                    raise RuntimeError(f"tensor id {p.ps_attr.grad_id()}")

        if args.use_hybrid_adam:
            margin_chunk_num_for_gpu_adam = mgr.get_margin_chunk_num_for_gpu_adam(
            )
        else:
            margin_chunk_num_for_gpu_adam = 0

        max_chunk_size = self.client.chunk_list.max_chunk_size()
        self.read_chunk_buff = FP32ChunkReadBuffer(
            self.client.chunk_list, self.client.chunk_tensor_index,
            max_chunk_size, margin_chunk_num_for_gpu_adam)
        self.write_chunk_buff = FP16ChunkWriteBuffer(
            self.client.chunk_list, self.client.chunk_tensor_index,
            max_chunk_size)

        # 混合ADMA，根据预热获得的信息，放一部分Chunk在GPU上。
        self.FP16_f_adamv2(self.client, fp32_param_list,
                           fp16_param_with_grad_list, exp_avgs, exp_avg_sqs,
                           max_exp_avg_sqs, state_steps, False, beta1_list,
                           beta2_list, lr_list, weight_decay_list, eps_list,
                           self.prefer_device, self.read_chunk_buff,
                           self.write_chunk_buff, True,
                           margin_chunk_num_for_gpu_adam)

        global_timer.my_timer.finish_profile('ADAM')
        mgr = PatrickStarManager()

        if mgr.is_warmup_training():
            logger.info('******** SHOW ACCESS INFO ********')
            for idx, chunk in self.client.chunk_list.generate_chunk():
                chunk.display_access_mom_info()
        mgr.reset_metronome()

        return loss
