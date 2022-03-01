# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from copy import deepcopy
import math
from typing import List
import torch
import time

from patrickstar.core.const import TensorState, TrainingStage
from patrickstar.core.hook import reduce_grad
from patrickstar.core.parameter import ParamType
import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import logger, get_rank
from .op_builder.cpu_adam import CPUAdamBuilder
from patrickstar.profiler import profiler


class FP16Adam(torch.optim.Optimizer):
    optimizer_id = 0

    def __init__(
        self,
        client,
        params,
        loss_scaler=None,
        gradient_clipping=-1,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        use_adamw=False,
        amsgrad=False,
    ):
        """
        The implementation was based on
        https://github.com/pytorch/pytorch/blob/c371542efc/torch/optim/optimizer.py
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(FP16Adam, self).__init__(params, defaults)
        self.client = client

        self.loss_scaler = loss_scaler
        self.has_overflow = False

        self.gradient_clipping = gradient_clipping
        # clamp_scalar_cpu does not support fp16. Turn the gradient_clipping
        # to tensor to use clamp_cpu instead.
        self.cpu_gradient_clipping = torch.Tensor([gradient_clipping])

        assert (
            len(self.param_groups) == 1
        ), "Only support one param group at the moment."
        # Eager state initialization, different from Pytorch
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                self.state[p]["betas"] = group["betas"]
                self.state[p]["lr"] = group["lr"]
                self.state[p]["weight_decay"] = group["weight_decay"]
                self.state[p]["eps"] = group["eps"]

                state["step"] = 0

                # Only create the local optimizer state params.
                state["exp_avg"] = torch.zeros(
                    p.ps_attr.shape, device=torch.device("cpu:0")
                )
                state["exp_avg_sq"] = torch.zeros(
                    p.ps_attr.shape, device=torch.device("cpu:0")
                )

        self.use_adamw = use_adamw
        self.opt_id = FP16Adam.optimizer_id
        FP16Adam.optimizer_id = FP16Adam.optimizer_id + 1
        try:
            # The pre-compiled cpu adam extension.
            from .adam import cpu_adam_op
        except ImportError:
            cpu_adam_op = CPUAdamBuilder().load()

        self.ds_opt_adam = cpu_adam_op
        self.ds_opt_adam.create_adam(
            self.opt_id,
            lr,
            betas[0],
            betas[1],
            eps,
            weight_decay,
            self.use_adamw,
            True,
        )

    def __del__(self):
        # need to destroy the C++ object explicitly to avoid a memory leak when intialize_engine
        # is used multiple times in the same process.
        self.ds_opt_adam.destroy_adam(self.opt_id)

    def __setstate__(self, state):
        super(FP16Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def ds_cpu_adam_update(
        self,
        data,
        grad,
        momentum,
        variance,
        step,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction,
    ):
        """
        This function will update the data, momentum and variance inplace.
        """
        assert data.device.type == "cpu"
        assert grad.device.type == "cpu"
        assert momentum.device.type == "cpu"
        assert variance.device.type == "cpu"

        loss_scale = self.loss_scaler.loss_scale if self.loss_scaler is not None else -1
        # Inputs of DS CPU Adam need to be flattened.
        self.ds_opt_adam.adam_update(
            self.opt_id,
            step,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction,
            data.view(-1),
            grad.view(-1),
            momentum.view(-1),
            variance.view(-1),
            loss_scale,
        )

    def torch_adam_update(
        self,
        data,
        grad,
        exp_avg,
        exp_avg_sq,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2,
    ):
        if self.loss_scaler is not None:
            grad.div_(self.loss_scaler.loss_scale)
        if weight_decay != 0:
            if self.use_adamw:
                # Perform stepweight decay
                data.mul_(1 - lr * weight_decay)
            else:
                grad = grad.add(data, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # TODO(jiaruifang) dose not support amsgrad
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        data.addcdiv_(exp_avg, denom, value=-step_size)

    def check_overflow(self, param):
        if (
            self.loss_scaler is not None
            and not self.has_overflow
            and self.loss_scaler.has_overflow(param)
        ):
            self.has_overflow = True

    def has_overflow_and_reset_param(self):
        r"""Method for collective communicating overflow and reset params.
        This method should be called after checking if each individual
        grad has overflow.
        """
        if torch.distributed.is_initialized():
            overflow_gpu = torch.cuda.ByteTensor([self.has_overflow])
            torch.distributed.all_reduce(
                overflow_gpu, op=torch.distributed.ReduceOp.MAX, async_op=False
            )
            self.has_overflow = overflow_gpu[0].item()
        return self.has_overflow

    def fp16_chunk_adam_ops(
        self,
        fp16_param_with_grad_list,
        exp_avg_list: List[torch.nn.Parameter],
        exp_avg_sq_list: List[torch.nn.Parameter],
        state_steps: List[int],
        amsgrad: bool,
        hyperparam_list: List[dict],
        time_profile=True,
    ):
        r"""Functional API that performs Adam algorithm computation.
        Visit fp16_param_with_grad_list in the order of tensors stored in chunks.
        Copy the chunk into a tmp buffer to speed up the memcpy between devices.
        """
        for i, param in enumerate(fp16_param_with_grad_list):
            # 1. prepare data for Adam
            if time_profile:
                global_timer.my_timer.start_profile("ADAM_prepare_data")

            grad_tensor = param.grad
            grad_tensor = grad_tensor.to(torch.device("cpu"))

            # Gradient clipping
            if self.gradient_clipping > 0:
                # The gradient clipping may be larger than the max fp16 value
                # after being amplified by loss scale.
                max_fp16 = torch.finfo(torch.half).max
                if grad_tensor.device.type == "cpu":
                    gradient_clipping = self.cpu_gradient_clipping
                    if self.loss_scaler is not None:
                        gradient_clipping *= self.loss_scaler.loss_scale
                    gradient_clipping = min(torch.Tensor([max_fp16]), gradient_clipping)
                else:
                    gradient_clipping = self.gradient_clipping
                    if self.loss_scaler is not None:
                        gradient_clipping *= self.loss_scaler.loss_scale
                    gradient_clipping = min(max_fp16, gradient_clipping)
                grad_tensor.clamp_(-gradient_clipping, gradient_clipping)

            compute_device = grad_tensor.device

            # 2. Start Adam
            if time_profile:
                global_timer.my_timer.finish_profile("ADAM_prepare_data")
                global_timer.my_timer.start_profile("ADAM_compute")

            if param.ps_attr.param_type == ParamType.TORCH_BASED:
                fp32_data_tensor = param.data
            else:
                fp32_data_tensor = param.ps_attr.fp32
            exp_avg = exp_avg_list[i]
            exp_avg_sq = exp_avg_sq_list[i]

            step = state_steps[i]
            beta1, beta2 = hyperparam_list[i]["betas"]
            eps = hyperparam_list[i]["eps"]
            weight_decay = hyperparam_list[i]["weight_decay"]
            lr = hyperparam_list[i]["lr"]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if compute_device.type == "cpu" and grad_tensor.device.type == "cpu":
                self.ds_cpu_adam_update(
                    fp32_data_tensor,
                    grad_tensor,
                    exp_avg,
                    exp_avg_sq,
                    step,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    True,
                )
            else:
                fp32_grad_tensor = grad_tensor.float()
                self.torch_adam_update(
                    fp32_data_tensor,
                    fp32_grad_tensor,
                    exp_avg,
                    exp_avg_sq,
                    lr,
                    beta1,
                    beta2,
                    eps,
                    weight_decay,
                    bias_correction1,
                    bias_correction2,
                )

            if time_profile:
                global_timer.my_timer.finish_profile("ADAM_compute")
                global_timer.my_timer.start_profile("ADAM_param_fp32_to_fp16")

            # 3. Finish Adam.
            if time_profile:
                global_timer.my_timer.finish_profile("ADAM_param_fp32_to_fp16")

    def release_last(self):
        # the starting module would trigger post_module_backward hook
        rank = get_rank()
        for name, param in self.client.module.named_parameters():
            reduce_grad(param, self.client)
            if param.ps_attr.param_type == ParamType.TORCH_BASED:
                continue
            if param.ps_attr.get_state() == TensorState.COMPUTE:
                logger.debug(
                    f"adam forces rank {rank} to"
                    f"release param {self.client.module.__class__.__name__}.{name} from COMPUTE to HOLD_AFTER_BWD"
                )
                if torch.distributed.is_initialized():
                    self.client.release_dist(
                        param,
                        TensorState.HOLD_AFTER_BWD,
                        training_stage=TrainingStage.BWD,
                    )
                else:
                    self.client.release(param, TensorState.HOLD_AFTER_BWD)

    def init_fp16(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.ps_attr.param_type == ParamType.CHUNK_BASED:
                    data_fp16 = self.client.access(p, torch.device("cpu:0"))
                    data_fp32 = p.ps_attr.fp32
                    data_fp16.copy_(data_fp32)
                    self.client.release(p)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        global_timer.my_timer.start_profile("ADAM")
        self.release_last()
        if profiler.started():
            profiler.stage_convert_time.append((time.time(), TrainingStage.ADAM))

        self.client.set_training_phase(TrainingStage.ADAM)

        self.client.trigger_memory_tracing()
        self.client.adjust_chunk_layout()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.has_overflow_and_reset_param():
            global_timer.my_timer.finish_profile("ADAM")
            old_loss_scale = self.loss_scaler.loss_scale
            self.loss_scaler.update_scale(True)
            new_loss_scale = self.loss_scaler.loss_scale
            logger.warning(
                f"Gradient overflow! Update loss scale from {old_loss_scale} to {new_loss_scale}."
            )
            self.init_fp16()
            self.has_overflow = False
            return loss

        fp16_param_with_grad_list = []
        exp_avg_list = []
        exp_avg_sq_list = []

        hyperparam_list = []
        state_steps = []

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    # update the steps for each param group update
                    state = self.state[p]
                    state["step"] += 1

                    fp16_param_with_grad_list.append(p)

                    exp_avg_list.append(state["exp_avg"])
                    exp_avg_sq_list.append(state["exp_avg_sq"])

                    hyperparam = {
                        "betas": state["betas"],
                        "lr": state["lr"],
                        "weight_decay": state["weight_decay"],
                        "eps": state["eps"],
                    }

                    hyperparam_list.append(hyperparam)

                    # record the step after step update
                    state_steps.append(state["step"])

        print("before fp16_chunk_adam_ops")

        # Hybrid Adam. Put some chunks on GPU based on the warmup info.
        self.fp16_chunk_adam_ops(
            fp16_param_with_grad_list,
            exp_avg_list,
            exp_avg_sq_list,
            state_steps,
            False,
            hyperparam_list,
            True,
        )

        print("after fp16_chunk_adam_ops")

        if self.loss_scaler:
            self.loss_scaler.update_scale(False)

        self.init_fp16()

        global_timer.my_timer.finish_profile("ADAM")
        return loss

    def state_dict(self):
        r"""Returns the state of the optimizer as a :class:`dict`.

        It contains two entries:

        * state - a dict holding current optimization state. Its content
            differs between optimizer classes.
        * param_groups - a dict containing all parameter self.param_groups
        """
        raw_state_dict = super().state_dict()
        old_packed_state = raw_state_dict["state"]
        param_groups = raw_state_dict["param_groups"]
        assert len(param_groups) == 1
        packed_state = {}
        for idx in old_packed_state:
            packed_state[idx] = {}
            for k, v in old_packed_state[idx].items():
                if isinstance(v, torch.nn.Parameter):
                    packed_state[idx][k] = (
                        self.client.access(v, torch.device("cpu:0")).clone().detach()
                    )
                else:
                    packed_state[idx][k] = v
        return {
            "state": packed_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # deepcopy, to be consistent with module API
        state_dict = deepcopy(state_dict)

        saved_groups = state_dict["param_groups"]
        saved_state = state_dict["state"]

        assert len(saved_groups) == 1
        assert len(saved_groups[0]["params"]) == len(self.param_groups[0]["params"])
        assert len(saved_state) == len(self.state)

        # Update the state
        id_map = {
            old_id: p
            for old_id, p in zip(
                saved_groups[0]["params"], self.param_groups[0]["params"]
            )
        }

        for idx, p in id_map.items():
            saved_single_state = saved_state[idx]
            single_state = self.state[p]
            assert len(saved_single_state) == len(single_state)
            for k, v in single_state.items():
                if isinstance(v, torch.nn.Parameter):
                    tensor = self.client.access(v, torch.device("cpu:0"))
                    tensor.copy_(saved_single_state[k])
                else:
                    single_state[k] = saved_single_state[k]
