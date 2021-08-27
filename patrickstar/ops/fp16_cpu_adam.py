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
from patrickstar.fp16 import loss_scaler
import torch
import time
from pathlib import Path
from torch import Tensor
from typing import List, Optional
import logging

from patrickstar.core.const import PSTensorStatus, AccessType, TrainingStage
import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import print_rank, logger, get_sys_memory_used
from patrickstar.core.parameter import register_param, ParamType
from patrickstar.manager import PatrickStarManager
from patrickstar.core import ChunkList, ChunkTensorIndex, ChunkListType
from .chunk_io_buff import FP32ChunkReadBuffer, FP16ChunkWriteBuffer

from .op_builder import CPUAdamBuilder


def get_real_data_tensor(param):
    if param.ps_attr.param_type == ParamType.TORCH_BASED:
        return param.data
    elif param.ps_attr.param_type == ParamType.CHUNK_BASED:
        return param.ps_attr.access_tensor(AccessType.DATA)
    else:
        raise RuntimeError


class FP16Adam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(self,
                 client,
                 params,
                 loss_scaler=None,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0,
                 amsgrad=False,
                 prefer_device=torch.device('cpu:0'),
                 use_hybrid_adam: bool = True):
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

        self.loss_scaler = loss_scaler
        self.has_overflow = False

        self.prefer_device = prefer_device
        self.use_hybrid_adam = use_hybrid_adam
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

                if p.ps_attr.param_type == ParamType.TORCH_BASED:
                    if p.requires_grad:
                        # torch param 没有备份的 fp32 参数
                        state['fp32_param_data'] = None
                        state['exp_avg'] = torch.nn.Parameter(
                            torch.zeros_like(p,
                                             dtype=torch.float,
                                             device=torch.device('cpu:0')),
                            requires_grad=False)
                        register_param(state['exp_avg'], ParamType.TORCH_BASED,
                                       torch.float)
                        state['exp_avg_sq'] = torch.nn.Parameter(
                            torch.zeros_like(p,
                                             dtype=torch.float,
                                             device=torch.device('cpu:0')),
                            requires_grad=False)
                        register_param(state['exp_avg_sq'],
                                       ParamType.TORCH_BASED, torch.float)
                else:
                    name = p.ps_attr.name
                    state[
                        'fp32_param_data'] = self.client.param_fp16_to_param_fp32(
                            p)
                    state['exp_avg'] = torch.nn.Parameter(
                        torch.tensor(
                            [],
                            dtype=torch.float,
                            # memory_format=torch.preserve_format,
                            device=torch.device('cpu:0')),
                        requires_grad=False)
                    register_param(state['exp_avg'], ParamType.CHUNK_BASED,
                                   torch.float, f'{name}.exp_avg')
                    state['exp_avg'].ps_attr.reset_shape(p.ps_attr.shape)
                    state['exp_avg'].ps_attr._is_local = p.ps_attr.is_local()

                    state['exp_avg_sq'] = torch.nn.Parameter(
                        torch.tensor(
                            [],
                            dtype=torch.float,
                            # memory_format=torch.preserve_format,
                            device=torch.device('cpu:0')),
                        requires_grad=False)
                    register_param(state['exp_avg_sq'], ParamType.CHUNK_BASED,
                                   torch.float, f'{name}.exp_avg_sq')
                    state['exp_avg_sq'].ps_attr.reset_shape(p.ps_attr.shape)
                    state['exp_avg_sq'].ps_attr._is_local = p.ps_attr.is_local(
                    )

                    self.client.append_tensor(state['exp_avg'], torch.float,
                                              AccessType.DATA,
                                              ChunkListType.MOMENTUM,
                                              f'{name}_fp32')

                    self.client.append_tensor(state['exp_avg_sq'], torch.float,
                                              AccessType.DATA,
                                              ChunkListType.VARIANCE,
                                              f'{name}_fp32')

        # 用作fp16 grad 存储的buffer
        self.read_chunk_buff = None
        self.use_ds_adam = True
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

    def ds_cpu_adam_update(self, data, grad, momentum, variance, step, lr,
                           beta1, beta2, eps, weight_decay, bias_correction):
        """
        This function will update the data, momentum and variance inplace.
        """
        assert data.device.type == 'cpu'
        assert grad.device.type == 'cpu'
        assert momentum.device.type == 'cpu'
        assert variance.device.type == 'cpu'

        loss_scale = self.loss_scaler.loss_scale if self.loss_scaler is not None else -1
        # Inputs of DS CPU Adam need to be flattened.
        self.ds_opt_adam.adam_update(self.opt_id, step, lr, beta1, beta2, eps,
                                     weight_decay, bias_correction,
                                     data.view(-1), grad.view(-1),
                                     momentum.view(-1), variance.view(-1),
                                     loss_scale)

    def torch_adam_update(self, data, grad, exp_avg, exp_avg_sq, lr, beta1,
                          beta2, eps, weight_decay, bias_correction1,
                          bias_correction2):
        if self.loss_scaler is not None:
            grad.div_(self.loss_scaler.loss_scale)
        if weight_decay != 0:
            grad = grad.add(data, alpha=weight_decay)

        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if False:  # amsgrad
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs, exp_avg_sq, out=max_exp_avg_sqs)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs.sqrt() /
                     math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        data.addcdiv_(exp_avg, denom, value=-step_size)

    def check_overflow(self, param):
        if (self.loss_scaler is not None and not self.has_overflow
                and self.loss_scaler.has_overflow(param)):
            self.has_overflow = True

    def has_overflow_and_reset_param(self, write_chunk_buff):
        """
        这个函数应该在已经判断过本进程是否存在 overflow 之后调用
        """
        if torch.distributed.is_initialized():
            overflow_gpu = torch.cuda.ByteTensor([self.has_overflow])
            torch.distributed.all_reduce(overflow_gpu,
                                         op=torch.distributed.ReduceOp.MAX)
            self.has_overflow = overflow_gpu[0].item()
        if self.has_overflow:
            # TODO(zilinzhu): Find a better way to overwrite the grads
            for _, p in self.client.module.named_parameters():
                if p.ps_attr.param_type == ParamType.TORCH_BASED:
                    continue
                if not self.client.is_local_tensor(p, AccessType.DATA):
                    continue
                fp32_param = self.state[p]['fp32_param_data']
                write_chunk_buff.write_from_cache(p, fp32_param)
            self.loss_scaler.update_scale(self.has_overflow)
            self.has_overflow = False
            write_chunk_buff.reset()
            return True
        return False

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
        local_rank = client.local_rank
        logger.info(
            f'local_rank {local_rank} margin_chunk_num_for_gpu_adam {margin_chunk_num_for_gpu_adam}, '
            f'param cnt {len(fp32_params)}')
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
            if fp16_param.ps_attr.param_type == ParamType.TORCH_BASED:
                # 如果fp16_param被Torch管理，则它肯定在cpu上，cpu_embedding优化引起的
                assert fp16_param.data.device.type == 'cpu'
                fp32_param = fp16_param
                # 这里已经是 fp32 的了
                fp16_grad_tensor = fp16_param.grad
                assert fp16_grad_tensor.dtype == torch.float
            else:
                # 将FP16 GPU Chunk拷贝到compute_device的FP32 Chunk上。
                # 如果是第一个tensor则拷贝Chunk，否则索引chunk
                fp16_grad_tensor = read_chunk_buff.access_from_cache(
                    fp16_param).view(fp16_param.ps_attr.shape)

            compute_device = fp16_grad_tensor.device

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
                self.ds_cpu_adam_update(fp32_data_tensor, fp16_grad_tensor,
                                        exp_avg, exp_avg_sq, step, lr, beta1,
                                        beta2, eps, weight_decay, True)
            else:
                fp32_grad_tensor = fp16_grad_tensor.float()
                self.torch_adam_update(fp32_data_tensor, fp32_grad_tensor,
                                       exp_avg, exp_avg_sq, lr, beta1, beta2,
                                       eps, weight_decay, bias_correction1,
                                       bias_correction2)

            if time_profile:
                global_timer.my_timer.finish_profile('ADAM_compute')
                global_timer.my_timer.start_profile('ADAM_param_fp32_to_fp16')

            ##########################
            ####### 结束ADAM计算 ######
            ##########################

            # Note fp16_param对应的Chunk内存 ->fp32_param对应的chunk内存
            if fp32_param.ps_attr.param_type == ParamType.CHUNK_BASED:
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
        rank = torch.distributed.get_rank()

        for name, param in self.client.module.named_parameters():
            if param.ps_attr.param_type == ParamType.TORCH_BASED:
                continue
            if param.ps_attr.get_status(
                    AccessType.DATA) == PSTensorStatus.COMPUTE:
                logger.debug(
                    f'rank {rank} release param {name} from COMPUTE to HOLD_AFTER_BWD'
                )
                tmp_tensor = param.ps_attr.access_tensor(AccessType.DATA)
                tmp_tensor.copy_(param.grad)
                param.grad = None

                if torch.distributed.is_initialized():
                    self.client.release_dist(param,
                                             AccessType.DATA,
                                             PSTensorStatus.HOLD_AFTER_BWD,
                                             training_stage=TrainingStage.BWD,
                                             is_allreduce=True)
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

        if self.use_hybrid_adam:
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

        if self.has_overflow_and_reset_param(
                write_chunk_buff=self.write_chunk_buff):
            global_timer.my_timer.finish_profile('ADAM')
            return loss

        max_param_size = 0
        for i, group in enumerate(self.param_groups):
            for j, p in enumerate(group['params']):
                if p.requires_grad:
                    # update the steps for each param group update
                    state = self.state[p]
                    state['step'] += 1

                    # p不是torch param，且p属于remote chunk跳过
                    if p.ps_attr.param_type == ParamType.CHUNK_BASED and not self.client.is_local_tensor(
                            p, AccessType.DATA):
                        continue

                    if p.ps_attr.param_type == ParamType.TORCH_BASED:
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

        if self.loss_scaler:
            self.loss_scaler.update_scale(self.has_overflow)

        return loss
