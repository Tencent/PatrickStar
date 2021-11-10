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

from patrickstar.core import ChunkType
from patrickstar.core.const import TensorState, AccessType, TrainingStage
from patrickstar.core.parameter import register_param, ParamType
from patrickstar.manager import RuntimeMemTracer
import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import logger, get_rank
from .chunk_io_buff import FP32ChunkReadBuffer, FP16ChunkWriteBuffer
from .op_builder.cpu_adam import CPUAdamBuilder
from patrickstar.utils.helper import get_real_data_tensor
from patrickstar.profiler import profiler


def zero_param(p):
    return torch.nn.Parameter(
        torch.zeros_like(p, dtype=torch.float),
        requires_grad=False,
    )


def empty_cpu_param():
    return torch.nn.Parameter(
        torch.tensor([], dtype=torch.float, device=torch.device("cpu:0")),
        requires_grad=False,
    )


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
        use_hybrid_adam=True,
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

        self.use_hybrid_adam = use_hybrid_adam

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

                if p.ps_attr.param_type == ParamType.TORCH_BASED:
                    if p.requires_grad:
                        state["exp_avg"] = zero_param(p)
                        register_param(
                            state["exp_avg"], ParamType.TORCH_BASED, torch.float
                        )
                        state["exp_avg_sq"] = zero_param(p)
                        register_param(
                            state["exp_avg_sq"], ParamType.TORCH_BASED, torch.float
                        )
                elif p.ps_attr.is_local():
                    # Only create the local optimizer state params.
                    name = p.ps_attr.name
                    state["exp_avg"] = empty_cpu_param()
                    register_param(
                        state["exp_avg"],
                        ParamType.CHUNK_BASED,
                        torch.float,
                        f"{name}.exp_avg",
                    )
                    state["exp_avg"].ps_attr.reset_shape(p.ps_attr.shape)
                    state["exp_avg"].ps_attr._is_local = p.ps_attr.is_local()

                    state["exp_avg_sq"] = empty_cpu_param()
                    register_param(
                        state["exp_avg_sq"],
                        ParamType.CHUNK_BASED,
                        torch.float,
                        f"{name}.exp_avg_sq",
                    )
                    state["exp_avg_sq"].ps_attr.reset_shape(p.ps_attr.shape)
                    state["exp_avg_sq"].ps_attr._is_local = p.ps_attr.is_local()

                    # Chunk layout of Momentum and Variance should be consist with param fp16
                    self.client.append_tensor_as_ref(
                        state["exp_avg"],
                        torch.float,
                        AccessType.DATA,
                        ChunkType.MOMENTUM,
                        p,
                    )

                    self.client.append_tensor_as_ref(
                        state["exp_avg_sq"],
                        torch.float,
                        AccessType.DATA,
                        ChunkType.VARIANCE,
                        p,
                    )

        # The buffer for fp16 grad.
        self.read_chunk_buff = None
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

    def has_overflow_and_reset_param(self, write_chunk_buff):
        r"""Method for collective communicating overflow and reset params.
        This method should be called after checking if each individual
        grad has overflow.
        """
        if torch.distributed.is_initialized():
            overflow_gpu = torch.cuda.ByteTensor([self.has_overflow])
            torch.distributed.all_reduce(
                overflow_gpu, op=torch.distributed.ReduceOp.MAX
            )
            self.has_overflow = overflow_gpu[0].item()
        if self.has_overflow:
            # TODO(zilinzhu): Find a better way to overwrite the grads
            for _, p in self.client.module.named_parameters():
                if p.ps_attr.param_type == ParamType.TORCH_BASED:
                    continue
                if not p.ps_attr.is_local():
                    continue
                fp32_param = self.client.param_fp16_to_param_fp32_map[p]
                write_chunk_buff.write_from_cache(p, fp32_param)
            self.has_overflow = False
            write_chunk_buff.reset()
            return True
        return False

    def fp16_chunk_adam_ops(
        self,
        client,
        fp32_param_list: List[torch.nn.Parameter],
        fp16_param_with_grad_list,
        exp_avg_list: List[torch.nn.Parameter],
        exp_avg_sq_list: List[torch.nn.Parameter],
        state_steps: List[int],
        amsgrad: bool,
        hyperparam_list: List[dict],
        read_chunk_buff,
        write_chunk_buff,
        time_profile=True,
        margin_chunk_num_for_gpu_adam=0,
    ):
        r"""Functional API that performs Adam algorithm computation.
        Visit fp16_param_with_grad_list in the order of tensors stored in chunks.
        Copy the chunk into a tmp buffer to speed up the memcpy between devices.
        """
        local_rank = client.local_rank
        logger.info(
            f"local_rank {local_rank} margin_chunk_num_for_gpu_adam {margin_chunk_num_for_gpu_adam}, "
            f"param cnt {len(fp32_param_list)}"
        )
        for i, fp32_param in enumerate(fp32_param_list):
            # 1. prepare data for Adam
            fp16_param = fp16_param_with_grad_list[i]

            if time_profile:
                global_timer.my_timer.start_profile("ADAM_prepare_data")
                global_timer.my_timer.start_profile("ADAM_prepare_data_grad_copy")

            # Copy the fp16 grads in the granularity of chunks.
            if fp16_param.ps_attr.param_type == ParamType.TORCH_BASED:
                # If fp16_param is managed by native torch, it should be on CPU,
                # because only cpu_embedding optimization are managed by native torch
                # now and it is fp32.
                assert fp32_param is None
                fp32_param = fp16_param
                # Here the grad is already of dtype fp32.
                fp16_grad_tensor = fp16_param.grad
                assert fp16_grad_tensor.dtype == torch.float
            else:
                # Copy the fp16 grad chunk to the compute_device of fp32 param chunk.
                # As we are visiting  params by its storing order in the chunk,
                # we will only copy the chunk when visiting its first tensor and store it
                # in the buffer. For the rest of the tensors, we will directly indexing from
                # the buffer.
                fp16_grad_tensor = read_chunk_buff.access_from_cache(fp16_param).view(
                    fp16_param.ps_attr.shape
                )

            # Gradient clipping
            if self.gradient_clipping > 0:
                # The gradient clipping may be larger than the max fp16 value
                # after being amplified by loss scale.
                max_fp16 = torch.finfo(torch.half).max
                if fp16_grad_tensor.device.type == "cpu":
                    gradient_clipping = self.cpu_gradient_clipping
                    if self.loss_scaler is not None:
                        gradient_clipping *= self.loss_scaler.loss_scale
                    gradient_clipping = min(torch.Tensor([max_fp16]), gradient_clipping)
                else:
                    gradient_clipping = self.gradient_clipping
                    if self.loss_scaler is not None:
                        gradient_clipping *= self.loss_scaler.loss_scale
                    gradient_clipping = min(max_fp16, gradient_clipping)
                fp16_grad_tensor.clamp_(-gradient_clipping, gradient_clipping)

            compute_device = fp16_grad_tensor.device

            if time_profile:
                global_timer.my_timer.finish_profile("ADAM_prepare_data_grad_copy")
                global_timer.data_move_cnter.update(
                    "ADAM_prepare_data_grad_copy", fp16_grad_tensor.numel() * 2
                )

            client.access_data(fp32_param, compute_device)
            fp32_data_tensor = get_real_data_tensor(fp32_param)

            exp_avg_param = exp_avg_list[i]
            exp_avg_sq_param = exp_avg_sq_list[i]

            client.access_data(exp_avg_param, compute_device)
            client.access_data(exp_avg_sq_param, compute_device)

            exp_avg = get_real_data_tensor(exp_avg_param)
            exp_avg_sq = get_real_data_tensor(exp_avg_sq_param)

            # 2. Start Adam
            if time_profile:
                global_timer.my_timer.finish_profile("ADAM_prepare_data")
                global_timer.my_timer.start_profile("ADAM_compute")

            step = state_steps[i]
            beta1, beta2 = hyperparam_list[i]["betas"]
            eps = hyperparam_list[i]["eps"]
            weight_decay = hyperparam_list[i]["weight_decay"]
            lr = hyperparam_list[i]["lr"]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            if compute_device.type == "cpu" and fp16_grad_tensor.device.type == "cpu":
                self.ds_cpu_adam_update(
                    fp32_data_tensor,
                    fp16_grad_tensor,
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
                fp32_grad_tensor = fp16_grad_tensor.float()
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

            # Copy fp32_param back to fp16_param.
            if fp32_param.ps_attr.param_type == ParamType.CHUNK_BASED:
                write_chunk_buff.write_from_cache(fp16_param, fp32_param)

            if time_profile:
                global_timer.my_timer.finish_profile("ADAM_param_fp32_to_fp16")
                global_timer.data_move_cnter.update(
                    "ADAM_param_fp32_to_fp16", fp32_data_tensor.numel() * 4
                )
                global_timer.my_timer.start_profile("ADAM_release_data")

            client.release_data(fp32_param)
            client.release_data(exp_avg_param)
            client.release_data(exp_avg_sq_param)

            if time_profile:
                global_timer.my_timer.finish_profile("ADAM_release_data")

        write_chunk_buff.reset()
        read_chunk_buff.reset()

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        global_timer.my_timer.start_profile("ADAM")

        rank = get_rank()
        mgr = RuntimeMemTracer()
        for name, param in self.client.module.named_parameters():
            if param.ps_attr.param_type == ParamType.TORCH_BASED:
                continue
            if param.ps_attr.get_state(AccessType.DATA) == TensorState.COMPUTE:
                self.client.optimizer.check_overflow(param)
                logger.debug(
                    f"adam forces rank {rank} to"
                    f"release param {self.client.module.__class__.__name__}.{name} from COMPUTE to HOLD_AFTER_BWD"
                )
                tmp_tensor = param.ps_attr.access_tensor(AccessType.DATA)
                tmp_tensor.copy_(param.grad)
                param.grad = None
                if torch.distributed.is_initialized():
                    self.client.release_dist(
                        param,
                        AccessType.DATA,
                        TensorState.HOLD_AFTER_BWD,
                        training_stage=TrainingStage.BWD,
                        is_allreduce=True,
                    )
                else:
                    self.client.release_data(param, TensorState.HOLD_AFTER_BWD)
        if profiler.started():
            profiler.stage_convert_time.append((time.time(), TrainingStage.ADAM))
        self.client.metronome.set_training_phase(TrainingStage.ADAM)
        mgr.tiktac(self.client)

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.use_hybrid_adam:
            margin_chunk_num_for_gpu_adam = mgr.get_margin_chunk_num_for_gpu_adam()
        else:
            margin_chunk_num_for_gpu_adam = 0

        max_chunk_size = self.client.chunk_list.max_chunk_size()
        self.read_chunk_buff = FP32ChunkReadBuffer(
            self.client.chunk_list,
            self.client.chunk_tensor_index,
            max_chunk_size,
            margin_chunk_num_for_gpu_adam,
        )
        self.write_chunk_buff = FP16ChunkWriteBuffer(
            self.client.chunk_list, self.client.chunk_tensor_index, max_chunk_size
        )

        if self.has_overflow_and_reset_param(write_chunk_buff=self.write_chunk_buff):
            global_timer.my_timer.finish_profile("ADAM")
            old_loss_scale = self.loss_scaler.loss_scale
            self.loss_scaler.update_scale(True)
            new_loss_scale = self.loss_scaler.loss_scale
            logger.warning(
                f"Gradient overflow! Update loss scale from {old_loss_scale} to {new_loss_scale}."
            )

            return loss

        fp16_param_with_grad_list = []
        fp32_param_list = []
        exp_avg_list = []
        exp_avg_sq_list = []

        hyperparam_list = []
        state_steps = []

        max_param_size = 0
        for _, group in enumerate(self.param_groups):
            for _, p in enumerate(group["params"]):
                if p.requires_grad:
                    # update the steps for each param group update
                    state = self.state[p]
                    state["step"] += 1

                    # When p is not torch param and belongs to a remote chunk, skip.
                    if (
                        p.ps_attr.param_type == ParamType.CHUNK_BASED
                        and not p.ps_attr.is_local()
                    ):
                        continue

                    if p.ps_attr.param_type == ParamType.TORCH_BASED:
                        max_param_size = max(p.numel(), max_param_size)

                    fp16_param_with_grad_list.append(p)

                    exp_avg_list.append(state["exp_avg"])
                    exp_avg_sq_list.append(state["exp_avg_sq"])
                    if p in self.client.param_fp16_to_param_fp32_map:
                        fp32_param_list.append(
                            self.client.param_fp16_to_param_fp32_map[p]
                        )
                    else:
                        fp32_param_list.append(None)
                    hyperparam = {
                        "betas": state["betas"],
                        "lr": state["lr"],
                        "weight_decay": state["weight_decay"],
                        "eps": state["eps"],
                    }

                    hyperparam_list.append(hyperparam)

                    # record the step after step update
                    state_steps.append(state["step"])

        # Hybrid Adam. Put some chunks on GPU based on the warmup info.
        self.fp16_chunk_adam_ops(
            self.client,
            fp32_param_list,
            fp16_param_with_grad_list,
            exp_avg_list,
            exp_avg_sq_list,
            state_steps,
            False,
            hyperparam_list,
            self.read_chunk_buff,
            self.write_chunk_buff,
            True,
            margin_chunk_num_for_gpu_adam,
        )

        if self.loss_scaler:
            self.loss_scaler.update_scale(False)

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
                        self.client.access_data(v, torch.device("cpu:0"))
                        .clone()
                        .detach()
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
                    tensor = self.client.access_data(v, torch.device("cpu:0"))
                    tensor.copy_(saved_single_state[k])
                else:
                    single_state[k] = saved_single_state[k]
