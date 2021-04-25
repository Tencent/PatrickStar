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
from torch.utils.data import SequentialSampler
from checkpoint import reset_checkpointed_activations_memory_buffer, checkpoint, init_checkpointed_activations_memory_buffer


class SimpleCKPModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleCKPModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def init_ckp(self, batch_size, data_type: torch.dtype):
        numel = (self.hidden_dim * batch_size) * 2
        init_checkpointed_activations_memory_buffer(numel, data_type)

    def _checkpointed_forward(self, x):
        """Forward method with activation checkpointing."""
        def custom():
            def custom_forward(x):
                x = self.linear1(x)
                x = self.linear2(x)
                return x

            return custom_forward

        # Make sure memory is freed.
        reset_checkpointed_activations_memory_buffer()
        x = checkpoint(custom(), x)
        return x

    def forward(self, x, y):
        x = self._checkpointed_forward(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return self.cross_entropy_loss(x, y)


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleModel, self).__init__()
        self.linear1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList(
                [torch.nn.Linear(hidden_dim, hidden_dim)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = x
        hidden_dim = self.linear1(hidden_dim)
        hidden_dim = self.linear2(hidden_dim)
        hidden_dim = self.linear3(hidden_dim)
        hidden_dim = self.linear4(hidden_dim)
        return self.cross_entropy_loss(hidden_dim, y)


def get_data_loader(batch_size,
                    total_samples,
                    hidden_dim,
                    device,
                    data_type=torch.float):
    train_data = torch.randn(total_samples,
                             hidden_dim,
                             device=device,
                             dtype=data_type)
    train_label = torch.empty(total_samples, dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader
