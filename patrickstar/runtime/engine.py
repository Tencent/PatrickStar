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

from patrickstar.utils import logger, global_timer
from patrickstar.utils import print_rank as print_rank_0
from patrickstar.core import PatrickStarClient, AccessType, PSChunkStatus, PSTensorStatus, TrainingStage
from patrickstar.manager import PatrickStarManager
from patrickstar.ops import FP16Adam
from patrickstar.fp16 import LossScaler, DynamicLossScaler


class PatrickStarEngine(Module):
    r"""DeepSpeed engine for training.
    """
    def __init__(self, model, client, config):
        super(PatrickStarEngine, self).__init__()
        self.module = model
        self.module.train()

        self.client = client

        prefer_device = torch.device(f'cpu:0')

        if client.local_rank == 0:
            logger.info(f'ADAM on device {prefer_device}')

        if config is not None:
            # Optimizer configuration
            optim_config = config["optimizer"]
            optim_type = optim_config["type"]
            if optim_type not in ["Adam", "AdamW"]:
                raise ValueError(f"Only support Adam and AdamW at the moment. "
                                 f"Get optimizer type {optim_type}")
            optim_params = optim_config["params"]

            # Loss scaler configuration
            if "fp16" not in config:
                self.loss_scaler = None
            else:
                loss_scale_config = config["fp16"]
                assert loss_scale_config[
                    "enabled"], "Must enable fp16 training."
                loss_scale = loss_scale_config["loss_scale"]
                if loss_scale == 0:
                    self.loss_scaler = DynamicLossScaler(
                        init_scale=2**loss_scale_config["initial_scale_power"],
                        scale_factor=loss_scale_config["hysteresis"],
                        scale_window=loss_scale_config["loss_scale_window"],
                        min_scale=loss_scale_config["min_loss_scale"])
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
                "use_hybrid_adam": True
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
            use_hybrid_adam=optim_params["use_hybrid_adam"])
        # 这个hook并没啥意义，为何不能和postbwd hook一起？
        # self.create_reduce_and_remove_grad_hooks()

        self.client.init(self.module, self.optimizer)
        logger.info('init PatrickStarEngine')

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        pass
        # print_rank_0(f"reduce_ready_partitions_and_remove_grads param {i}",
        #              force=True)
        # # self.reduce_independent_p_g_buckets_and_remove_grads(param, i)
        # print_rank_0(f'name {param.ps_attr.name} {param.grad}',
        #              force=True)
        # reduce grad and release grad，TODO(jiaruifang)确认这个hook和bwd hook的关系

    def create_reduce_and_remove_grad_hooks(self):
        # print_rank_0(f'[Begin] Create gradient reduction hooks', force=True)
        self.grad_accs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            for param in param_group['params']:
                if param.requires_grad:
                    #print_rank_0(f" Before all gather {param.device}, {param.shape}")

                    # The hook must be created in un-partitioned parameter
                    # param.all_gather()

                    #print(f"After all gather {param.device}, {param.shape}")
                    def wrapper(param, i):
                        param_tmp = param.expand_as(param)
                        grad_acc = param_tmp.grad_fn.next_functions[0][0]

                        def reduce_partition_and_remove_grads(*notneeded):
                            self.reduce_ready_partitions_and_remove_grads(
                                param, i)

                        grad_acc.register_hook(
                            reduce_partition_and_remove_grads)
                        self.grad_accs.append(grad_acc)

                    print(
                        f"param grad fn {param.expand_as(param).grad_fn.next_functions[0][0]}"
                    )
                    wrapper(param, i)
                    # print_rank_0(f"warp param on group {i}", force=True)

                    # Partition the parameter after creating the hook
                    # param.partition()
        print_rank_0(f'[End] Create gradient reduction hooks', force=True)

    def forward(self, *inputs, **kwargs):
        r"""Execute forward propagation
        Arguments:
            *inputs: Variable length input list
            **kwargs: variable length keyword arguments
        """
        global_timer.my_timer.start_profile("FWD")
        mgr = PatrickStarManager()
        mgr._training_stage = TrainingStage.FWD

        for param_fp16 in self.client.chunk_based_param_fp16:
            param_fp16.ps_attr.fwd_used_cnt = 0

        loss = self.module(*inputs, **kwargs)
        for chunk_id, chunk in self.client.chunk_list.generate_chunk():
            if chunk.get_status() == PSChunkStatus.HOLD_AFTER_FWD:
                self.client.set_all_tensors_status_in_chunk(
                    chunk_id, PSTensorStatus.HOLD)
        global_timer.my_timer.finish_profile("FWD")
        return loss

    def backward(self, loss, allreduce_gradients=True, release_loss=False):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: is deprecated, ignored, and will soon be removed'
        """
        global_timer.my_timer.start_profile("BWD")
        mgr = PatrickStarManager()
        mgr._training_stage = TrainingStage.BWD
        for param_fp16 in self.client.chunk_based_param_fp16:
            param_fp16.ps_attr.bwd_used_cnt = 0

        self.optimizer.zero_grad()
        if self.loss_scaler:
            self.loss_scaler.backward(loss)
        else:
            loss.backward()
        mgr.update_margin_mem()
        global_timer.my_timer.finish_profile("BWD")
