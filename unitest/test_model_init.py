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
from patrickstar import PatrickStarManager
from patrickstar.core import PatrickStarClient, ChunkTensorIndex, ChunkList, AccessType, ParamType
import logging
import torch
from common import distributed_test
from transformers import BertModel, BertConfig


class TestModelInitContext(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[2], backend="gloo", use_fake_dist=True)
    def test_model_init(self):
        mgr = PatrickStarManager(0)

        def model_provider():
            cfg = BertConfig()
            cfg.vocab_size = 10
            model = BertModel(cfg)
            return model

        compute_device = torch.device('cpu:0')
        default_chunk_size = 32 * 1024 * 1024
        client = PatrickStarClient(0, default_chunk_size, is_fp16=True)

        torch.manual_seed(0)
        with PSPreProcessCtx(client, dtype=torch.float, use_fake_dist=True):
            ps_model = model_provider()

        torch.manual_seed(0)
        torch_model = model_provider()

        for ps_param, torch_param in zip(ps_model.parameters(),
                                         torch_model.parameters()):
            if ps_param.ps_attr.param_type == ParamType.TORCH_BASED:
                self.assertLess(
                    torch.max(torch_param.data - ps_param), 1e-4,
                    f"PyTorch tensors are not consist with each other")
            else:
                ps_data = client.access_data(ps_param, compute_device)
                if ps_param.ps_attr.is_local():
                    self.assertLess(
                        torch.max(torch_param.data - ps_data), 1e-4,
                        f"{ps_param.ps_attr.name} ps tensor and pytorch tensor are not consist with each other"
                    )
                client.release_data(ps_param)

        # client.chunk_tensor_index.visit_chunks(client.chunk_list)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    unittest.main()
