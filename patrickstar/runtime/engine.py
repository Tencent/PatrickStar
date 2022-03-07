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

from patrickstar.core import (
    TrainingStage,
    ParamType,
    register_param,
)
from patrickstar.ops import FP16Adam
from patrickstar.utils import log_dist, global_timer
from patrickstar.core.hook import setup_patrickstar_hooks
from patrickstar.core.const import TensorState
from patrickstar.core.hook import reduce_grad
from .checkpoint import state_dict, load_state_dict


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
            },
        }

        if config is not None:
            # Optimizer configuration
            optim_config = config.get("optimizer", default_optim_config)
            optim_type = optim_config.get("type", default_optim_config["type"])
            if optim_type != "Adam":
                raise ValueError(
                    f"Only support Adam and AdamW at the moment. "
                    f"Get optimizer type {optim_type}"
                )
            optim_params = optim_config.get("params", default_optim_config["params"])
            for key, val in default_optim_config["params"].items():
                if key not in optim_params:
                    optim_params[key] = val
        else:
            optim_type = default_optim_config["type"]
            optim_params = default_optim_config["params"]

        params = list(self.parameters())
        if len(params) == 0:
            dummy = torch.nn.Parameter(torch.empty([1]), requires_grad=False)
            register_param(dummy, ParamType.TORCH_BASED, "dummy")
            params = [dummy]

        self.optimizer = FP16Adam(
            self.client,
            params,
            lr=optim_params["lr"],
            betas=optim_params["betas"],
            eps=optim_params["eps"],
            weight_decay=optim_params["weight_decay"],
        )

        self.client.optimizer = self.optimizer
        self.client.module = self.module
        setup_patrickstar_hooks(model, self.client)

        # This need to be placed before the initialization of optimizer.
        self.move_torch_parts_to_gpu(self.module)

        self.iteration_cnt_ = 0
        log_dist("PatrickStarEngine initialized.")

    def move_torch_parts_to_gpu(self, model):
        # TODO(zilinzhu) Currently we move all buffers to GPU as the buffer size is
        # relatively small. Maybe find a better way to deal with them.
        for buffer in model.buffers():
            buffer.data = buffer.data.to(self.client.device)

        def move_param_to_gpu(module):
            for param in module.parameters(recurse=False):
                if param.ps_attr.param_type == ParamType.TORCH_BASED:
                    param.data = param.data.to(self.client.device)
            for submodule in module.children():
                move_param_to_gpu(submodule)

        move_param_to_gpu(model)

    def _reset_before_forward(self):
        self.client.mem_tracer.reset_memory_stats()
        self.client.mem_tracer.metronome.reset()
        for chunk in self.client.chunk_list.chunks:
            chunk.unused = 0

    def parameters(self, recurse: bool = True):
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        for name, param in self.module.named_parameters(prefix, recurse):
            if param.ps_attr.is_local():
                yield name, param

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
        if self.iteration_cnt_ == 1:
            self.client.set_warmup(False)
            self.client.mem_tracer.end()

        global_timer.start_profile("FWD")

        self.client.set_training_stage(TrainingStage.FWD)
        self._reset_before_forward()

        loss = self.module(*inputs, **kwargs)
        global_timer.finish_profile("FWD")
        return loss

    def release_last(self):
        # the starting module would not trigger post_module_backward hook
        flag = False
        for param in self.client.module.parameters():
            reduce_grad(param, self.client)
            if param.ps_attr.param_type == ParamType.TORCH_BASED:
                continue
            if param.ps_attr.state == TensorState.COMPUTE:
                self.client.release(param)
                flag = True
        if flag:
            self.client.mem_tracer.trace()

    def backward(self, loss, scaler=None):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
        """
        global_timer.start_profile("BWD")
        self.client.set_training_stage(TrainingStage.BWD)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        self.release_last()
        self.iteration_cnt_ += 1
        global_timer.finish_profile("BWD")

    def step(self, scaler=None):
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.ps_attr.grad is not None:
                    self.optimizer.client.access(p, torch.device("cpu:0"))
                    p.grad = p.ps_attr.grad
                    p.ps_attr.grad = None
        if scaler is not None:
            scaler.step(self.optimizer)
        else:
            self.optimizer.step()
        for group in self.optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self.optimizer.client.release(p)
                    p.grad = None

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
