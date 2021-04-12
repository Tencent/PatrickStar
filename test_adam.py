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

import os
import json
import argparse
import torch
from torch.utils.data import SequentialSampler
import torch.optim as optim
from cpu_adam import CPUAdam
from client import HybridPSClient
from manager import HybridPSManager
from hook import setup_hybrid_ps_hooks
import logging
import time


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = x
        hidden_dim = self.linear(hidden_dim)
        hidden_dim = self.linear2(hidden_dim)
        hidden_dim = self.linear3(hidden_dim)
        hidden_dim = self.linear4(hidden_dim)
        return self.cross_entropy_loss(hidden_dim, y)


def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = 4  #model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=torch.float)
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader


def show_optim(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            print(p.size())


def test_simple_model(is_ps: False):
    hidden_dim = 4
    device = torch.device('cuda:0')

    model = SimpleModel(hidden_dim, empty_grad=False)
    model.cuda()

    if is_ps:
        logging.info('test a simple model with hybrid ps')

    data_loader = get_data_loader(model=model,
                                  total_samples=1000,
                                  hidden_dim=hidden_dim,
                                  device=device)

    loss_res = []
    if is_ps:
        client = HybridPSClient(gpu_index=0, default_chunk_size=20)
        optimizer = CPUAdam(client, model.parameters(), lr=0.001)
        client.register_module(model)
        setup_hybrid_ps_hooks(model, client)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_time = time.time()
    for n, batch in enumerate(data_loader):
        optimizer.zero_grad()
        loss = model(batch[0], batch[1])
        # if torch.distributed.get_rank() == 0:
        print("LOSS:", loss.item())
        loss_res.append(loss.item())
        loss.backward()

        optimizer.step()

        if is_ps:
            client.release_all_grad()
        if n == 5: break

    elapse = time.time() - start_time
    logging.info(f"is_ps {is_ps} elapse {elapse}")
    return loss_res


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)
    torch.manual_seed(0)
    manager = HybridPSManager()
    # 4 layer每层20个elem，最少360内存
    manager.init([40] * 1, [280])

    loss_ref_list = test_simple_model(False)

    torch.manual_seed(0)
    loss_list = test_simple_model(True)

    for loss, loss_ref in zip(loss_list, loss_ref_list):
        assert loss == loss_ref
