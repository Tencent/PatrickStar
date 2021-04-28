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
from utils import see_memory_usage
import contexttimer


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

    def _bandwidth_benchmark(self, size, niter):
        buff = torch.zeros(size)
        # for i in range(niter):
        #   buff.append(torch.zeros(size))
        with contexttimer.Timer() as t:
            # for i in range(niter):
            buff.to(self.compute_device)
        BWD = size * 4 / t.elapsed / 1e9
        logging.info(f'size {size * 4/1024} KB, bandwidth {BWD} GB/s')

    def test_bandwidth(self):
        for size in [
                1024, 32 * 1024, 64 * 1024, 128 * 1024, 512 * 1024,
                1024 * 1024, 1024 * 1024 * 32, 1024 * 1204 * 1024
        ]:
            self._bandwidth_benchmark(size, 10)


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
