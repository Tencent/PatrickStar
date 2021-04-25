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
from tests.bert_classification import get_bert_data_loader
from tests.bert_classification import BertForSequenceClassification
# from tests.modeling import BertForSequenceClassification, BertConfig
from transformers import BertConfig
import enum
import time
import sys

from checkpoint import checkpoint
import logging
import torch
from utils import see_memory_usage
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
import time
import argparse

from client import HybridPSClient
from manager import HybridPSManager
from utils import setup_hybrid_ps_hooks
from ops import CPUAdam, TorchAdam

logging.basicConfig(
    format=
    '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

device = torch.device('cuda:0')

hidden_dim = 768
batch_size = 4
sequence_length = 256
cfg = BertConfig(gradient_checkpointing=False,
                 hidden_dim=hidden_dim,
                 num_hidden_layers=1)
# cfg = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
#         num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
torch.manual_seed(0)
model = BertForSequenceClassification(cfg)
model.cuda()


def show_grads(model, is_ps, step):
    print(f'show grads {step}')
    for name, param in model.named_parameters(recurse=True):
        print(
            name,
            torch.sum(param.grad)
            if not is_ps else torch.sum(param.ps_grad_tensor))


def show_params(model, is_ps, step):
    print(f'show params {step}')
    for name, param in model.named_parameters(recurse=True):
        print(
            name,
            torch.sum(param) if not is_ps else torch.sum(param.ps_data_tensor),
            param.shape, param.requires_grad)


data_loader = get_bert_data_loader(batch_size=batch_size,
                                   total_samples=1000,
                                   sequence_length=sequence_length,
                                   device=device,
                                   data_type=torch.float)

loss_res = []

manager = HybridPSManager()
manager.init([1024 * 1024 * 512] * 1, [1024 * 1024 * 1024 * 4 * 4])
# chunk 32 M
client = HybridPSClient(gpu_index=0, default_chunk_size=1024 * 1024 * 16)
optimizer = TorchAdam(model.parameters(), lr=0.001)

client.register_module(model)
setup_hybrid_ps_hooks(model, client)
# optimizer = TorchAdam(model.parameters(), lr=0.001)
optimizer = CPUAdam(client, model.parameters(), lr=0.001)

start_time = time.time()
for n, batch in enumerate(data_loader):
    logging.info(
        f'input sum {torch.sum(batch[0])}, label sum {torch.sum(batch[1])}')
    # show_params(model, False, n)

    output = model(input_ids=batch[0], labels=batch[1])
    loss = output[0]
    # logits = output.logits
    # if torch.distributed.get_rank() == 0:
    print(f"LOSS: {loss.item()}")
    print("*" * 20)
    loss_res.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    print("*" * 20)
    # show_grads(model, False, n)
    print("*" * 20)
    # show_params(model, False, n)

    break
