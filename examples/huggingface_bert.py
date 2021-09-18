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
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

from patrickstar.runtime import initialize_engine

from lmdb_dataset import get_dataset


# Uncomment this line when doing multiprocess training
# torch.distributed.init_process_group(backend='nccl')

train_dataset, _, _ = get_dataset("aclImdb")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def model_func():
    return BertForSequenceClassification.from_pretrained("bert-base-uncased")


LR = 5e-5
BETAS = (0.9, 0.999)
EPS = 1e-6
WEIGHT_DECAY = 0

config = {
    # The same format as optimizer config of DeepSpeed
    # https://www.deepspeed.ai/docs/config-json/#optimizer-parameters
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": LR,
            "betas": BETAS,
            "eps": EPS,
            "weight_decay": WEIGHT_DECAY,
            "use_hybrid_adam": True,
        },
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "initial_scale_power": 2 ** 3,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "default_chunk_size": 64 * 1024 * 1024,
    "release_after_init": False,
    "use_cpu_embedding": False,
}

model, optim = initialize_engine(model_func=model_func, local_rank=0, config=config)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for epoch in range(3):
    for i, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids = batch["input_ids"].to(device)
        # print(input_ids)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        model.backward(loss)
        optim.step()
        print(i, loss.item())
        if i == 10:
            exit()

model.eval()
