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

import unittest

import torch
from torch.nn import Embedding as TorchEmbedding
from transformers import BertConfig

from common import distributed_test
from patrickstar.ops import Embedding as PSEmbedding


class TestClientAccess(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_embedding(self):
        cfg = BertConfig()
        cfg.hidden_dropout_prob = 0
        test_device = torch.device("cuda:0")
        seq_len = 10
        torch.manual_seed(0)
        input_ids = torch.randint(
            low=0,
            high=cfg.vocab_size - 1,
            size=(1, seq_len),
            dtype=torch.long,
            device=test_device,
        )

        torch.manual_seed(0)
        torch_embedding = TorchEmbedding(cfg.vocab_size, 64)
        torch.manual_seed(0)
        PSEmbedding.use_cpu = True
        ps_embedding = PSEmbedding(cfg.vocab_size, 64)

        res = ps_embedding(input_ids)
        torch_res = torch_embedding.to(test_device)(input_ids)

        self.assertLess(torch.max(torch_res.cpu() - res.cpu()), 1e-2)


if __name__ == "__main__":
    unittest.main()
