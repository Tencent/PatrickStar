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
from patrickstar.core.preprocess import PSPreProcessCtx
from patrickstar.core import PatrickStarClient, ChunkTensorIndex, ChunkList
import logging
import torch
from tests.simple_net import SimpleModel
from patrickstar.utils import init_distributed
from patrickstar.deepspeed_helper.global_vars import set_global_variables
from common import distributed_test


class TestModelInitContext(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_model_init(self):
        def model_provider():
            return SimpleModel(12, False, False)

        chunk_tensor_index = ChunkTensorIndex()
        chunkmgr = ChunkList()
        client = PatrickStarClient(0, 1000, is_fp16=True)
        with PSPreProcessCtx(chunk_tensor_index, chunkmgr, client):
            model_provider()


if __name__ == "__main__":
    set_global_variables()
    unittest.main()
