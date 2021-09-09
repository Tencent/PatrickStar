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

import torch
from transformers import BertModel, BertConfig

from common import distributed_test
from patrickstar.core import PSPreProcessCtx
from patrickstar.core import PatrickStarClient
from patrickstar.ops import FP16Adam


class TestModelInitContext(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_optimizer_init(self):
        def model_provider():
            cfg = BertConfig()
            cfg.vocab_size = 10
            model = BertModel(cfg)
            return model

        default_chunk_size = 32 * 1024 * 1024
        client = PatrickStarClient(0, default_chunk_size, is_fp16=True)

        torch.manual_seed(0)
        with PSPreProcessCtx(client, dtype=torch.float):
            ps_model = model_provider()

        FP16Adam(client, ps_model.parameters())


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    unittest.main()
