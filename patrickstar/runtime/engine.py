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

import torch

from patrickstar.core import ChunkState, TensorState, TrainingStage, ParamType
from patrickstar.fp16 import LossScaler, DynamicLossScaler
from patrickstar.ops import FP16Adam
from patrickstar.utils import log_dist, global_timer

from .checkpoint import state_dict, load_state_dict
from patrickstar.profiler import profiler
import time


class PatrickStarEngine(torch.nn.Module):
    r"""patrickStar engine for training."""

    def __init__(self, model, client, config):
        super(PatrickStarEngine, self).__init__()
        self.module = model
        self.module.train()

        self.client = client

        # default parameter for adam.
        default_optim_config = {
            "type": "Adam",
            "params": {
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "use_hybrid_adam": True,
            },
        }

        if config is not None:
            # Optimizer configuration
            optim_config = config.get("optimizer", default_optim_config)
            optim_type = optim_config.get("type", default_optim_config["type"])
            if optim_type not in ["Adam", "AdamW"]:
                raise ValueError(
                    f"Only support Adam and AdamW at the moment. "
                    f"Get optimizer type {optim_type}"
                )
            optim_params = optim_config.get("params", default_optim_config["params"])
            for key, val in default_optim_config["params"].items():
                if key not in optim_params:
                    optim_params[key] = val

            # Loss scaler configuration
            if "fp16" not in config:
                self.loss_scaler = None
            else:
                loss_scale_config = config["fp16"]
                assert loss_scale_config["enabled"], "Must enable fp16 training."
                assert (
                    "loss_scale" in loss_scale_config
                ), "Must have `loss_scale` field set."
                loss_scale = loss_scale_config["loss_scale"]
                if loss_scale == 0:
                    log_dist("Use DynamicLossScaler")
                    self.loss_scaler = DynamicLossScaler(
                        init_scale=(
                            2 ** loss_scale_config.get("initial_scale_power", 16)
                        ),
                        scale_factor=loss_scale_config.get("hysteresis", 2),
                        scale_window=loss_scale_config.get("loss_scale_window", 2000),
                        min_scale=loss_scale_config.get("min_loss_scale", 1),
                    )
                else:
                    self.loss_scaler = LossScaler(loss_scale)

            # Gradient clipping configuration
            if "gradient_clipping" not in config:
                self.gradient_clipping = -1
            else:
                self.gradient_clipping = config["gradient_clipping"]
        else:
            optim_type = default_optim_config["type"]
            optim_params = default_optim_config["params"]
            self.loss_scaler = None
            self.gradient_clipping = -1

        # This need to be placed before the initialization of optimizer.
        self._move_torch_parts_to_gpu(model)

        self.optimizer = FP16Adam(
            self.client,
            self.module.parameters(),
            loss_scaler=self.loss_scaler,
            gradient_clipping=self.gradient_clipping,
            lr=optim_params["lr"],
            betas=optim_params["betas"],
            eps=optim_params["eps"],
            weight_decay=optim_params["weight_decay"],
            use_adamw=(optim_type == "AdamW"),
            use_hybrid_adam=optim_params["use_hybrid_adam"],
        )

        self.client.init(self.module, self.optimizer)
        self.iteration_cnt_ = 0
        # TODO(jiaruifang) pass in via config.
        self.warmup_times = 1
        log_dist("PatrickStarEngine initialized.")

    def _move_torch_parts_to_gpu(self, model):
        # TODO(zilinzhu) Currently we move all buffers to GPU as the buffer size is
        # relatively small. Maybe find a better way to deal with them.
        for buffer in model.buffers():
            buffer.data = buffer.data.to(self.client.device)

        def move_param_to_gpu(module):
            if module.__class__.__name__ == "Embedding":
                return
            for param in module.parameters(recurse=False):
                if param.ps_attr.param_type == ParamType.TORCH_BASED:
                    param.data = param.data.to(self.client.device)
            for submodule in module.children():
                move_param_to_gpu(submodule)

        move_param_to_gpu(model)

    def _reset_before_forward(self):
        # TODO(jiaruifang) so difficult to understand.
        # about grad overflow.
        self.client.mem_tracer.reset_memory_stats()
        self.client.mem_tracer.metronome.reset()
        for param_fp16 in self.client.chunk_based_param_fp16:
            param_fp16.ps_attr.fwd_used_cnt = 0
        for _, chunk in self.client.chunk_list.generate_chunk():
            chunk.unused = 0

        self.client.reset_visited_chunk()

    def _set_state_after_forward(self):
        """
        After forward calculation, we need to reset the state of
        tensors from HOLD_AFTER_FWD to HOLD. Otherwise, chunks may be
        released accidentally when using gradient checkpointing.
        """
        for chunk_id, chunk in self.client.chunk_list.generate_chunk():
            if (
                chunk.get_state() == ChunkState.HOLD
                or chunk.get_state() == ChunkState.HOLD_AFTER_FWD
            ):
                chunk.set_unused()
                self.client.set_all_tensors_state_in_chunk(chunk_id, TensorState.HOLD)

    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation
        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        # warmup logic, we have to make sure a iteration run the entire FWD+BWD process.
        # Considering the grad overflow situation.
        if self.iteration_cnt_ == 0:
            self.client.set_warmup(True)
        if self.iteration_cnt_ == self.warmup_times:
            self.client.set_warmup(False)
            self.client.mem_tracer.close_tracer()

        global_timer.my_timer.start_profile("FWD")
        if profiler.started():
            profiler.stage_convert_time.append((time.time(), TrainingStage.FWD))

        self.client.set_training_phase(TrainingStage.FWD)
        self._reset_before_forward()

        loss = self.module(*inputs, **kwargs)
        self._set_state_after_forward()
        global_timer.my_timer.finish_profile("FWD")
        self.client.reset_visited_chunk()
        return loss

    def backward(self, loss):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
        """
        global_timer.my_timer.start_profile("BWD")
        if profiler.started():
            profiler.stage_convert_time.append((time.time(), TrainingStage.FWD))
        self.client.set_training_phase(TrainingStage.BWD)

        for param_fp16 in self.client.chunk_based_param_fp16:
            param_fp16.ps_attr.bwd_used_cnt = 0

        self.optimizer.zero_grad()
        if self.loss_scaler:
            self.loss_scaler.backward(loss)
        else:
            loss.backward()
        self.client.mem_tracer.update_margin_mem()
        self.iteration_cnt_ += 1
        global_timer.my_timer.finish_profile("BWD")

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return state_dict(
            self.module,
            self.client,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

    def load_state_dict(self, state_dict, strict=False):
        return load_state_dict(
            self.module, self.client, state_dict=state_dict, strict=strict
        )
