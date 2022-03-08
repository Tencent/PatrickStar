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

from common import distributed_test
from patrickstar.core import PatrickStarClient, PSPreProcessCtx, torch_scope, ParamType


class TestTorchScopeContext(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_torch_scope(self):
        def model_provider():
            with torch_scope():
                return torch.nn.Linear(5, 10)

        default_chunk_size = 1 * 1024 * 1024
        client = PatrickStarClient(0, default_chunk_size)

        with PSPreProcessCtx(client, dtype=torch.float):
            ps_model = model_provider()

        assert ps_model.weight.ps_attr.param_type == ParamType.TORCH_BASED


if __name__ == "__main__":
    unittest.main()
