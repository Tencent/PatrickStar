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
import logging
import unittest

import torch
from transformers import BertConfig, BertForSequenceClassification

from common import distributed_test
from examples.data_loader import get_bert_data_loader
from patrickstar.runtime import initialize_engine
from patrickstar.utils import logger

logger.setLevel(logging.WARNING)


def test_bert_model(
    method,
    batch_size=32,
    hidden_dim=768,
    sequence_length=512,
    num_layer=12,
    num_head=12,
):
    # Avoid gpu0 use more memory.
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    rank = torch.distributed.get_rank()
    torch.cuda.empty_cache()

    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    cfg = BertConfig(
        hidden_size=hidden_dim,
        intermediate_size=hidden_dim * 4,
        max_position_embeddings=sequence_length,
        num_attention_heads=num_head,
        num_hidden_layers=num_layer,
    )

    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-6
    weight_decay = 0

    def model_func():
        return BertForSequenceClassification(cfg)

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
                "use_hybrid_adam": True,
            },
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 10,
            "loss_scale_window": 1000,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "default_chunk_size": 32 * 1024 * 1024,
        "release_after_init": True,
        "use_cpu_embedding": True,
    }

    model, optimizer = initialize_engine(
        model_func=model_func, local_rank=rank, config=config
    )

    model.eval()
    torch.save(model.state_dict(), "test_checkpoint.pt")

    data_loader = get_bert_data_loader(
        batch_size=batch_size,
        total_samples=10000,
        sequence_length=sequence_length,
        device=device,
        is_distrbuted=True,
    )

    batch0 = next(iter(data_loader))

    output = model(input_ids=batch0[0], labels=batch0[1])
    loss_before = output[0].item()
    print("loss:", loss_before)

    model.train()
    for n, batch in enumerate(data_loader):
        if n == 5:
            break
        optimizer.zero_grad()

        output = model(input_ids=batch[0], labels=batch[1])
        loss = output[0]
        model.backward(loss)
        optimizer.step()
    model.eval()

    output = model(input_ids=batch0[0], labels=batch0[1])
    loss_trained = output[0].item()
    print("loss after several training steps:", loss_trained)

    state_dict = torch.load("test_checkpoint.pt")
    model.load_state_dict(state_dict)

    output = model(input_ids=batch0[0], labels=batch0[1])
    loss_after = output[0].item()
    print("loss after load checkpoint:", loss_after)

    assert loss_trained != loss_before
    assert loss_before == loss_after


class TestModelInitContext(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1], backend="gloo", use_fake_dist=False)
    def test_checkpoint(self):
        # 0.11B
        hidden_dim = 768
        sequence_length = 512
        num_layer = 6
        num_head = 12

        batch_size = 2

        assert hidden_dim % num_head == 0

        torch.manual_seed(0)
        test_bert_model(
            method="patrickstar",
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_layer=num_layer,
            num_head=num_head,
        )


if __name__ == "__main__":
    unittest.main()
