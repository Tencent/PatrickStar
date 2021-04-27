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
from client import HybridPSClient, ChunkList, PSTensorStatus, AccessType
import logging
import torch
from manager import HybridPSManager


class TestAccess(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 20
        self.client = HybridPSClient(
            gpu_index=0, default_chunk_size=self.default_chunk_size)
        self.manager = HybridPSManager()
        self.manager.init([32, 32], [1024])
        self.compute_device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info('SetUp finished')

    def test_register(self):
        self.manager.reset([40 * 4], [256 * 4])
        param1 = torch.nn.Parameter(torch.zeros(23, dtype=torch.float))
        assert param1.requires_grad is True
        self.client.visit()
        # self.client.register_param(param1)
        # assert param1.data_status == PSTensorStatus.HOLD
        # assert param1.grad_status == PSTensorStatus.FREE

    def test_access(self):
        self.manager.reset([40 * 4], [256 * 4])
        param1 = torch.nn.Parameter(torch.zeros(10, dtype=torch.float))

        assert param1.requires_grad is True

        logging.info(f'access param1 data')
        self.client.register_param(param1)
        self.client.access_data(param1, self.compute_device)
        assert param1.data_status == PSTensorStatus.COMPUTE
        assert param1.grad_status == PSTensorStatus.FREE
        assert self.client.get_chunk_id(param1, AccessType.DATA) == 0

        self.client.visit()
        logging.info(f'access param1 grad')
        self.client.access_grad(param1, self.compute_device)
        assert param1.data_status == PSTensorStatus.COMPUTE
        assert param1.grad_status == PSTensorStatus.COMPUTE
        assert self.client.get_chunk_id(param1, AccessType.GRAD) == 0

        logging.info(f'access param2 data')
        param2 = torch.nn.Parameter(torch.zeros(10, dtype=torch.float))
        self.client.register_param(param2)

        logging.info(f'release param1 grad')
        self.client.release_grad(param1, PSTensorStatus.FREE)

        # 测试复用
        logging.info(f'release param2 grad')
        self.client.access_grad(param2, self.compute_device)

        self.client.visit()


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
