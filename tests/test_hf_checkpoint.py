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


class TestHfCheckpointContext(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_hf_checkpoint(self):
        hidden_dim = 768
        sequence_length = 512
        num_layer = 6
        num_head = 12
        batch_size = 2
        # Avoid gpu0 use more memory.
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
        rank = torch.distributed.get_rank()

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

        hf_model = BertForSequenceClassification(cfg)
        hf_model.eval()
        hf_model.to(device)

        def model_func():
            return BertForSequenceClassification(cfg)

        config = {
            # The same format as optimizer config of DeepSpeed
            # https://www.deepspeed.ai/docs/config-json/#optimizer-parameters
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": 0.001,
                    "betas": (0.9, 0.999),
                    "eps": 1e-6,
                    "weight_decay": 0,
                },
            },
            "fp16": {
                "loss_scale": "dynamic",
                "init_scale": 2 ** 8,
            },
            "default_chunk_size": 32 * 1024 * 1024,
            "release_after_init": True,
        }

        model, _ = initialize(model_func=model_func, local_rank=rank, config=config)
        model.eval()

        data_loader = get_bert_data_loader(
            batch_size=batch_size,
            total_samples=10000,
            sequence_length=sequence_length,
            device=device,
        )
        batch0 = next(iter(data_loader))

        output = hf_model(input_ids=batch0[0], labels=batch0[1])
        loss0 = output[0].item()
        print("loss of huggingface:", loss0)

        output = model(input_ids=batch0[0], labels=batch0[1])
        loss1 = output[0].item()
        print("loss of patrickstar:", loss1)
        self.assertTrue(abs(loss1 - loss0) > 1e-3, f"{loss1} vs {loss0}")

        model.load_state_dict(hf_model.state_dict())
        output = model(input_ids=batch0[0], labels=batch0[1])
        loss2 = output[0].item()
        print("loss of patrickstar with huggingface checkpoint:", loss2)
        # PatrickStar uses fp16, so there will be some difference.
        self.assertTrue(
            abs(loss2 - loss0) < 1e-3, f"{loss2} vs {loss0}, diff: {abs(loss2 - loss0)}"
        )

        hf_model.load_state_dict(model.state_dict())
        output = hf_model(input_ids=batch0[0], labels=batch0[1])
        loss3 = output[0].item()
        print("loss of huggingface with patrickstar checkpoint:", loss3)
        self.assertTrue(loss3 == loss0, f"{loss3} vs {loss0}")


if __name__ == "__main__":
    unittest.main()
