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

import random
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification, AdamW

from patrickstar.runtime import initialize_engine
from patrickstar.utils import logger
import logging
from simple_net import SimpleModel, get_bert_data_loader
device = torch.device('cuda') if torch.cuda.is_available() else torch.device(
    'cpu')

batch_size = 8
hidden_dim = 4
seq_len = 128


def model_func():
    return SimpleModel(hidden_dim=hidden_dim,
                       seq_len=seq_len,
                       is_ckp=True,
                       is_share_param=False)


lr = 5e-5
betas = (0.9, 0.999)
eps = 1e-6
weight_decay = 0

test_case = "torch"
logger.setLevel(logging.WARNING)

config = {
    # The same format as optimizer config of DeepSpeed
    # https://www.deepspeed.ai/docs/config-json/#optimizer-parameters
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "use_hybrid_adam": True
        }
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 2**3,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "default_chunk_size": 1024,
    "use_fake_dist": False,
    "use_cpu_embedding": False
}

torch.manual_seed(0)
if test_case == "patrickstar":
    model, optim = initialize_engine(model_func=model_func,
                                     local_rank=0,
                                     config=config)
elif test_case == "torch":
    model = model_func()
    optim = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             betas=betas,
                             eps=eps,
                             weight_decay=weight_decay)
    model.cuda()
else:
    raise RuntimeError

train_loader = get_bert_data_loader(batch_size, 10000, 128, device, False)

for epoch in range(3):
    for i, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids, labels = batch
        loss = model(input_ids, labels)
        if test_case == "patrickstar":
            model.backward(loss)
            optim.step()
        elif test_case == "torch":
            loss.backward()
            optim.zero_grad()
            optim.step()
        print(i, loss.item())
        if i == 10:
            exit()

model.eval()
