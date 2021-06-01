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

from utils import logger, init_distributed
from utils import print_rank as print_rank_0, debug_flag
from torch.nn.modules import Module
from client import HybridPSClient, AccessType
import torch
from ops import FP16Adam


class HybridPSEngine(Module):
    r"""DeepSpeed engine for training.
    """
    def __init__(self,
                 args,
                 model,
                 optimizer=None,
                 model_parameters=None,
                 training_data=None,
                 lr_scheduler=None,
                 mpu=None,
                 dist_init_required=None,
                 collate_fn=None,
                 config=None,
                 config_params=None,
                 dont_change_device=False):
        super(HybridPSEngine, self).__init__()
        if not torch.distributed.is_initialized():
            self.dist_backend = "gloo" if debug_flag else "nccl"
            init_distributed(dist_backend=self.dist_backend)

        # TODO 和args的local_rank什么关系？等价么
        self.rank = 0 if debug_flag else torch.distributed.get_rank()
        self.training_dataloader = None
        self.lr_scheduler = None
        self.module = model
        self.module.train()

        self.client = HybridPSClient(
            rank=self.rank,
            default_chunk_size=config.default_chunk_size,
            warmup=False,
            is_fp16=True)

        # TODO(jiaruifang) prefer_device应该是自适应的
        self.optimizer = FP16Adam(self.client,
                                  self.module.parameters(),
                                  lr=0.001,
                                  prefer_device=torch.device(f'cpu:0'))
        # prefer_device = torch.device(f'cuda:{self.rank}')
        # 这个hook并没啥意义，为何不能和postbwd hook一起？
        self.create_reduce_and_remove_grad_hooks()

        self.client.init(self.module, self.optimizer)
        logger.info('init HybridPSEngine')

        # for param in self.module.named_parameters():
        #     print(param)
        #creates backward hooks for gradient partitioning

    def reduce_ready_partitions_and_remove_grads(self, param, i):
        print_rank_0(f"reduce_ready_partitions_and_remove_grads param {i}",
                     force=True)
        # self.reduce_independent_p_g_buckets_and_remove_grads(param, i)
        print_rank_0(f'ps_name {param.ps_attr.ps_name} {param.grad}',
                     force=True)
        # reduce grad and release grad，TODO(jiaruifang)确认这个hook和bwd hook的关系

    def create_reduce_and_remove_grad_hooks(self):
        print_rank_0(f'[Begin] Create gradient reduction hooks', force=True)
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
        loss = self.module(*inputs, **kwargs)
        return loss

    def backward(self, loss, allreduce_gradients=True, release_loss=False):
        r"""Execute backward pass on the loss
        Arguments:
            loss: Torch tensor on which to execute backward propagation
            allreduce_gradients: is deprecated, ignored, and will soon be removed'
        """
        self.optimizer.zero_grad()
        loss.backward()
