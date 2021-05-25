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
from torch.nn.modules import Module
from client import HybridPSClient
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
        self.dist_backend = "nccl"
        init_distributed(dist_backend=self.dist_backend)

        # TODO 和args的local_rank什么关系？等价么
        self.rank = torch.distributed.get_rank()
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
        self.client.init(self.module, self.optimizer)
        logger.info('init HybridPSEngine')

        # for param in self.module.named_parameters():
        #     print(param)

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
