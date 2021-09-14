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

from common import distributed_test
from patrickstar import PatrickStarManager
from patrickstar.core import PatrickStarClient, AccessType, register_param, ChunkListType
from patrickstar.core.parameter import ParamType


class TestClientAccess(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40
        logging.info('SetUp finished')

    @distributed_test(world_size=[1])
    def test_append_ps_tensor(self):
        PatrickStarManager(0)
        self.client = PatrickStarClient(
            rank=0, default_chunk_size=self.default_chunk_size)

        self.compute_device = torch.device('cpu:0')

        param_size_list = [10, 11, 12, 13]

        param_list = []
        param_payload_ref_list = []
        for idx, psize in enumerate(param_size_list):
            param = torch.nn.Parameter(torch.rand(psize))
            param_list.append(param)
            param_payload_ref_list.append(param.data.clone())

            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{idx}")
            self.client.append_tensor([param], torch.float, AccessType.DATA,
                                      ChunkListType.PARAM_FP32)

            real_payload = self.client.access_data(param,
                                                   torch.device('cpu:0'))
            real_payload.copy_(param.data)
            self.client.release_data(param)
            self.assertTrue(param.data.numel() == 0)

        self.client.display_chunk_info()
        for param, payload_ref in zip(param_list, param_payload_ref_list):
            real_payload = self.client.access_data(param,
                                                   torch.device('cpu:0'))
            self.assertEqual(torch.max(real_payload - payload_ref), 0)
            self.client.release_data(param)

    @distributed_test(world_size=[1])
    def test_append_torch_tensor(self):
        self.client = PatrickStarClient(
            rank=0, default_chunk_size=self.default_chunk_size)

        self.compute_device = torch.device('cpu:0')

        param_size_list = [10, 11, 12, 13]

        param_list = []
        param_payload_ref_list = []
        for idx, psize in enumerate(param_size_list):
            param = torch.nn.Parameter(torch.rand(psize))
            param_list.append(param)
            register_param(param, ParamType.TORCH_BASED, torch.float,
                           f"param_{idx}")
            param_payload_ref_list.append(param.data.clone())
            self.client.append_tensor([param], torch.float, AccessType.DATA,
                                      ChunkListType.PARAM_FP32)

            real_payload = self.client.access_data(param,
                                                   torch.device('cpu:0'))
            real_payload.copy_(param.data)
            self.client.release_data(param)

        self.client.display_chunk_info()
        for param, payload_ref in zip(param_list, param_payload_ref_list):
            real_payload = self.client.access_data(param,
                                                   torch.device('cpu:0'))
            self.assertEqual(torch.max(real_payload - payload_ref), 0)
            self.client.release_data(param)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    unittest.main()
