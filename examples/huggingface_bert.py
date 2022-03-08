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
from patrickstar.utils import get_rank

from imdb_dataset import get_dataset


# Uncomment these lines when doing multiprocess training
# torch.distributed.init_process_group(backend='nccl')
# torch.cuda.set_device(get_rank())

train_dataset, _, test_dataset = get_dataset("/root/aclImdb")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def model_func():
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    # For large models, please uncomment the following lines to utilize gradient checkpointing
    # model.gradient_checkpointing_enable()
    return model


config = {
    # The same format as optimizer config of DeepSpeed
    # https://www.deepspeed.ai/docs/config-json/#optimizer-parameters
    "optimizer": {
        "type": "Adam",
        "params": {
            "lr": 5e-5,
            "betas": (0.9, 0.999),
            "eps": 1e-6,
            "weight_decay": 0,
        },
    },
    "chunk_size": 64 * 1024 * 1024,
    "release_after_init": False,
}

model, optim = initialize_engine(
    model_func=model_func, local_rank=get_rank(), config=config
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

print("train loss:")

for i, batch in enumerate(train_loader):
    optim.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    model.backward(loss)
    optim.step()
    print(i, loss.item())
    if i == 10:
        break

model.eval()

print("test loss:")

test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
for i, batch in enumerate(test_loader):
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    print(i, loss.item())
    if i == 5:
        break
