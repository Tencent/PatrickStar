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

import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from patrickstar.runtime import initialize_engine
from patrickstar.utils import get_rank

from examples.imdb_dataset import get_dataset
from moe_bert import build_moe_bert

parser = argparse.ArgumentParser()
parser.add_argument("--type", dest="type", type=str, choices=["patrickstar", "torch"])
parser.add_argument("--local_rank", dest="local_rank", type=int, default=None)
args = parser.parse_args()

torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(get_rank())

train_dataset, _, test_dataset = get_dataset("/root/aclImdb")

device = (
    torch.device(f"cuda:{get_rank()}")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

if args.type == "patrickstar":

    def model_func():
        return build_moe_bert()

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
        "release_after_init": True,
    }

    model, optim = initialize_engine(
        model_func=model_func, local_rank=args.local_rank, config=config
    )
else:
    model = build_moe_bert()
    optim = torch.optim.Adam(model.parameters(), lr=5e-5)
    model.cuda()


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

for batch in tqdm(train_loader):
    optim.zero_grad()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    if args.type == "patrickstar":
        model.backward(loss)
    else:
        loss.backward()
    optim.step()
