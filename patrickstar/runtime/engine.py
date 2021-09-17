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
import torch
from torch.nn.modules import Module

from patrickstar.core import PSChunkStatus, PSTensorStatus, TrainingStage
from patrickstar.fp16 import LossScaler, DynamicLossScaler
from patrickstar.manager import PatrickStarManager
from patrickstar.ops import FP16Adam
from patrickstar.utils import logger, global_timer

from .checkpoint import state_dict, load_state_dict


class PatrickStarEngine(Module):
    r"""DeepSpeed engine for training."""

    def __init__(self, model, client, config):
        super(PatrickStarEngine, self).__init__()
        self.module = model
        self.module.train()

        self.client = client

        prefer_device = torch.device("cpu:0")

        if client.local_rank == 0:
            logger.info(f"ADAM on device {prefer_device}")

        if config is not None:
            # Optimizer configuration
            optim_config = config["optimizer"]
            optim_type = optim_config["type"]
            if optim_type not in ["Adam", "AdamW"]:
                raise ValueError(
                    f"Only support Adam and AdamW at the moment. "
                    f"Get optimizer type {optim_type}"
                )
            optim_params = optim_config["params"]

            # Loss scaler configuration
            if "fp16" not in config:
                self.loss_scaler = None
            else:
                loss_scale_config = config["fp16"]
                assert loss_scale_config["enabled"], "Must enable fp16 training."
                loss_scale = loss_scale_config["loss_scale"]
                if loss_scale == 0:
                    self.loss_scaler = DynamicLossScaler(
                        init_scale=2 ** loss_scale_config["initial_scale_power"],
                        scale_factor=loss_scale_config["hysteresis"],
                        scale_window=loss_scale_config["loss_scale_window"],
                        min_scale=loss_scale_config["min_loss_scale"],
                    )
                else:
                    self.loss_scaler = LossScaler(loss_scale)

            # Gradient clipping configuration
            if "gradient_clipping" not in config:
                self.gradient_clipping = -1
            else:
                self.gradient_clipping = config["gradient_clipping"]
        else:
            # default parameter for adam.
            optim_type = "Adam"
            optim_params = {
                "lr": 0.01,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0,
                "use_hybrid_adam": True,
            }
            self.loss_scaler = None
            self.gradient_clipping = -1

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
            prefer_device=prefer_device,
            use_hybrid_adam=optim_params["use_hybrid_adam"],
        )

        self.client.init(self.module, self.optimizer)
        logger.info("init PatrickStarEngine")

    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation
        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        global_timer.my_timer.start_profile("FWD")
        mgr = PatrickStarManager()
        mgr.set_training_stage(TrainingStage.FWD)

        for param_fp16 in self.client.chunk_based_param_fp16:
            param_fp16.ps_attr.fwd_used_cnt = 0

        loss = self.module(*inputs, **kwargs)
        for chunk_id, chunk in self.client.chunk_list.generate_chunk():
            if chunk.get_status() == PSChunkStatus.HOLD_AFTER_FWD:
                self.client.set_all_tensors_status_in_chunk(
                    chunk_id, PSTensorStatus.HOLD
                )
        global_timer.my_timer.finish_profile("FWD")
        return loss

    def backward(self, loss):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
        """
        global_timer.my_timer.start_profile("BWD")
        mgr = PatrickStarManager()
        mgr.set_training_stage(TrainingStage.BWD)

        for param_fp16 in self.client.chunk_based_param_fp16:
            param_fp16.ps_attr.bwd_used_cnt = 0

        self.optimizer.zero_grad()
        if self.loss_scaler:
            self.loss_scaler.backward(loss)
        else:
            loss.backward()
        mgr.update_margin_mem()
        global_timer.my_timer.finish_profile("BWD")

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return state_dict(
            self,
            self.client,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )

    def load_state_dict(self, state_dict, strict=False):
        return load_state_dict(self, self.client, state_dict=state_dict, strict=strict)
