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

# from checkpoint.torch_checkpoint import checkpoint
from torch.utils.checkpoint import checkpoint
from torch.utils.data import SequentialSampler
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings


class Encoder(torch.nn.Module):
    def __init__(self, hidden_dim, is_ckp=False):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )

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


def get_data_loader(
    batch_size,
    total_samples,
    hidden_dim,
    device,
    data_type=torch.float,
    is_distrbuted=False,
):
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=data_type)
    train_label = torch.empty(total_samples, dtype=torch.long, device=device).random_(
        hidden_dim
    )
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    if is_distrbuted:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    return train_loader


def get_bert_data_loader(
    batch_size, total_samples, sequence_length, device, is_distrbuted=False
):
    train_data = torch.randint(
        low=0,
        high=10,
        size=(total_samples, sequence_length),
        device=device,
        dtype=torch.long,
    )
    train_label = torch.zeros(total_samples, dtype=torch.long, device=device)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    if is_distrbuted:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    return train_loader


class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, seq_len, is_ckp=False, is_share_param=False):
        super(SimpleModel, self).__init__()
        config = BertConfig()
        config.vocab_size = 25
        config.max_position_embeddings = seq_len
        config.hidden_size = hidden_dim
        self.embeddings_1 = BertEmbeddings(config)

        self._is_share_param = is_share_param
        if is_share_param:
            self.embeddings_2 = self.embeddings_1
        else:
            self.embeddings_2 = BertEmbeddings(config)
        self.encoder = Encoder(hidden_dim, is_ckp)
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        h1 = self.embeddings_1(x)
        h2 = self.embeddings_2(x)
        h3 = h1 + h2
        h3 = self.encoder(h3)
        return self.cross_entropy_loss(h3[:, 0], y)
