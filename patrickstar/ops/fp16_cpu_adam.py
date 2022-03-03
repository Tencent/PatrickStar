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


def adam(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    exp_avgs: List[torch.Tensor],
    exp_avg_sqs: List[torch.Tensor],
    max_exp_avg_sqs: List[torch.Tensor],
    state_steps: List[int],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1
        param.addcdiv_(exp_avg, denom, value=-step_size)


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
                    p.ps_attr.shape, device=torch.device("cuda:0")
                )
                state["exp_avg_sq"] = torch.zeros(
                    p.ps_attr.shape, device=torch.device("cuda:0")
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

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.loss_scaler is not None and self.has_overflow_and_reset_param():
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

        state_steps = []

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p.ps_attr.fp32)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            "Adam does not support sparse gradients, please consider SparseAdam instead"
                        )
                    grads.append(p.grad)

                    # update the steps for each param group update
                    state = self.state[p]

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=False,
            )

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
