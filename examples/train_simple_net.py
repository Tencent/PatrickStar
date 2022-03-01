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

import logging
import torch

from patrickstar.runtime import initialize_engine
from patrickstar.utils import logger

from simple_net import SimpleModel, get_bert_data_loader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

BATCH_SIZE = 8
HIDDEN_DIM = 4
SEQ_LEN = 128


def model_func():
    return SimpleModel(
        hidden_dim=HIDDEN_DIM, seq_len=SEQ_LEN, is_ckp=True, is_share_param=True
    )


LR = 5e-5
BETAS = (0.9, 0.999)
EPS = 1e-6
WEIGHT_DECAY = 0

# TEST_CASE = "torch"
TEST_CASE = "patrickstar"
logger.setLevel(logging.WARNING)
print(f"TEST_CASE {TEST_CASE}")
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
    "chunk_size": 1024,
    "use_fake_dist": False,
}

torch.manual_seed(0)
if TEST_CASE == "patrickstar":
    model, optim = initialize_engine(model_func=model_func, local_rank=0, config=config)
elif TEST_CASE == "torch":
    model = model_func()
    optim = torch.optim.Adam(
        model.parameters(), LR=LR, BETAS=BETAS, EPS=EPS, WEIGHT_DECAY=WEIGHT_DECAY
    )
    model.cuda()
else:
    raise RuntimeError

train_loader = get_bert_data_loader(BATCH_SIZE, 10000, 128, device, False)

for epoch in range(3):
    for i, batch in enumerate(train_loader):
        optim.zero_grad()
        input_ids, labels = batch
        loss = model(input_ids, labels)
        if TEST_CASE == "patrickstar":
            model.backward(loss)
            optim.step()
        elif TEST_CASE == "torch":
            loss.backward()
            optim.zero_grad()
            optim.step()
        print(i, loss.item())
        if i == 10:
            exit()

model.eval()
