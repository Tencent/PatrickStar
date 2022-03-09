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
from dataloader import get_bert_data_loader
from patrickstar.runtime import initialize
from patrickstar.utils import logger

logger.setLevel(logging.WARNING)


class TestCheckpointContext(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_checkpoint(self):
        # 0.11B
        hidden_dim = 768
        sequence_length = 512
        num_layer = 6
        num_head = 12

        batch_size = 2

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
            # Set dropout rate to 0 to prevent randomness in training.
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
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
                },
            },
            "default_chunk_size": 32 * 1024 * 1024,
            "release_after_init": True,
        }

        model, optimizer = initialize(
            model_func=model_func, local_rank=rank, config=config
        )

        data_loader = get_bert_data_loader(
            batch_size=batch_size,
            total_samples=10000,
            sequence_length=sequence_length,
            device=device,
        )
        batch0 = next(iter(data_loader))

        def train_one_step(batch):
            output = model(input_ids=batch[0], labels=batch[1])
            loss = output[0]
            model.backward(loss)
            model.step()
            model.zero_grad()

        # Train 5 steps first.
        for n, batch in enumerate(data_loader):
            if n == 5:
                break
            train_one_step(batch)

        # The loss after 5 steps.
        model.eval()
        output = model(input_ids=batch0[0], labels=batch0[1])
        loss0 = output[0].item()
        print("loss after the first 5 steps:", loss0)

        # Save checkpoints.
        rank = torch.distributed.get_rank()
        torch.save(model.state_dict(), f"model-{rank}.pt")
        torch.save(optimizer.state_dict(), f"optimizer-{rank}.pt")

        # Train 5 more steps and keep the data.
        batch_list = []
        model.train()
        for n, batch in enumerate(data_loader):
            if n == 5:
                break
            batch_list.append(batch)
            train_one_step(batch)

        # The loss after 10 steps.
        model.eval()
        output = model(input_ids=batch0[0], labels=batch0[1])
        loss1 = output[0].item()
        print("loss after 10 steps:", loss1)
        self.assertTrue(loss0 != loss1, "weight not updated")

        # Load checkpoint.
        model_state_dict = torch.load(f"model-{rank}.pt")
        opt_state_dict = torch.load(f"optimizer-{rank}.pt")
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(opt_state_dict)

        # The loss after checkpoint loading.
        model.eval()
        output = model(input_ids=batch0[0], labels=batch0[1])
        loss2 = output[0].item()
        print("loss after checkpoint loading:", loss2)

        self.assertTrue(loss0 == loss2, f"Model checkpoint error. {loss0} vs {loss2}")

        # Use the same data to train 5 steps.
        model.train()
        for batch in batch_list:
            train_one_step(batch)

        model.eval()
        output = model(input_ids=batch0[0], labels=batch0[1])
        loss3 = output[0].item()
        print("loss after checkpoint loading and 5 more training steps:", loss3)

        self.assertTrue(
            loss1 == loss3, f"Optimizer checkpoint error. {loss1} vs {loss3}"
        )


if __name__ == "__main__":
    unittest.main()
