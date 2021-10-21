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
import unittest

import torch
from transformers import BertConfig, BertForSequenceClassification

from common import distributed_test
from examples.data_loader import get_bert_data_loader
from patrickstar.runtime import initialize_engine
from patrickstar.utils import logger

logger.setLevel(logging.WARNING)


def bert_model(
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

    model, _ = initialize_engine(model_func=model_func, local_rank=rank, config=config)
    model.eval()

    data_loader = get_bert_data_loader(
        batch_size=batch_size,
        total_samples=10000,
        sequence_length=sequence_length,
        device=device,
        is_distrbuted=True,
    )
    batch0 = next(iter(data_loader))

    output = hf_model(input_ids=batch0[0], labels=batch0[1])
    loss0 = output[0].item()
    print("loss of huggingface:", loss0)

    output = model(input_ids=batch0[0], labels=batch0[1])
    loss1 = output[0].item()
    print("loss of patrickstar:", loss1)
    assert abs(loss1 - loss0) > 1e-3, f"{loss1} vs {loss0}"

    model.load_state_dict(hf_model.state_dict())
    output = model(input_ids=batch0[0], labels=batch0[1])
    loss2 = output[0].item()
    print("loss of patrickstar after load:", loss2)
    # PatrickStar uses fp16, so there will be some difference.
    assert abs(loss2 - loss0) < 1e-3, f"{loss2} vs {loss0}, diff: {abs(loss2 - loss0)}"

    hf_model.load_state_dict(model.state_dict())
    output = hf_model(input_ids=batch0[0], labels=batch0[1])
    loss3 = output[0].item()
    print("loss of huggingface after load:", loss3)
    assert loss3 == loss0, f"{loss3} vs {loss0}"


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

        bert_model(
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_layer=num_layer,
            num_head=num_head,
        )


if __name__ == "__main__":
    unittest.main()
