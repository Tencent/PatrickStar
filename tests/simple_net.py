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
# from checkpoint.torch_checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from utils import logger
from ops.embedding import ParallelBertEmbeddings, BertEmbeddings


class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim, is_ckp=False):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim))

        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear5 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.is_ckp = is_ckp

    def forward(self, x):
        h2 = self.linear1(x)
        if self.is_ckp:
            h3 = checkpoint(self.linear3, h2)
        else:
            h3 = self.linear3(h2)
        h4 = self.linear4(h3)
        h5 = self.linear5(h4)
        return h5


def get_data_loader(batch_size,
                    total_samples,
                    hidden_dim,
                    device,
                    data_type=torch.float,
                    is_distrbuted=False):
    train_data = torch.randn(total_samples,
                             hidden_dim,
                             device=device,
                             dtype=data_type)
    train_label = torch.empty(total_samples, dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    if is_distrbuted:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader


def get_bert_data_loader(batch_size,
                         total_samples,
                         sequence_length,
                         device,
                         is_distrbuted=False):
    train_data = torch.randint(low=0,
                               high=10,
                               size=(total_samples, sequence_length),
                               device=device,
                               dtype=torch.long)
    train_label = torch.zeros(total_samples, dtype=torch.long, device=device)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    if is_distrbuted:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, is_ckp=False, is_embed_opt=False):
        super(SimpleModel, self).__init__()

        if is_embed_opt:
            self.embeddings = ParallelBertEmbeddings(hidden_dim)
        else:
            self.embeddings = BertEmbeddings(hidden_dim)

        self.encoder = Encoder(hidden_dim, is_ckp)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        h1 = self.embeddings(x, device=x.device)
        h2 = self.encoder(h1)
        return self.cross_entropy_loss(h2[:, 0], y)
