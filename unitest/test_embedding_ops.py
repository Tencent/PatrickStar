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

import unittest
from torch.nn import Embedding as TorchEmbedding
from patrickstar.ops import Embedding as PSEmbedding
from patrickstar.utils import logger
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from common import distributed_test
import torch


class TestClientAccess(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_embedding(self):
        cfg = BertConfig()
        cfg.hidden_dropout_prob = 0
        test_device = torch.device('cuda:0')
        seq_len = 10
        torch.manual_seed(0)
        input_ids = torch.randint(low=0,
                                  high=cfg.vocab_size - 1,
                                  size=(1, seq_len),
                                  dtype=torch.long,
                                  device=test_device)

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
