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
from client import HybridPSClient, ChunkList, PSTensorStatus, AccessType, ChunkTensorIndex, Chunk
import logging
import torch
from manager import HybridPSManager
from client import PSChunkStatus


class TestChunkList(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40
        self.client = HybridPSClient(
            gpu_index=0, default_chunk_size=self.default_chunk_size)
        self.manager = HybridPSManager()
        self.manager.init([32 * 4], [1024])
        self.compute_device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def test_allocate(self):
        chunk_list = ChunkList()
        assert chunk_list.size() == 0

        chunk_list.new_chunk(chunk_id=0,
                             chunk_size=20,
                             data_type=torch.float,
                             compute_device=self.compute_device)
        assert chunk_list.size() == 1
        assert (chunk_list[0].get_status() == PSChunkStatus.RELEASED)

        chunk_list.access_chunk(0, self.compute_device)
        assert (chunk_list[0].get_status() == PSChunkStatus.FREE)

        chunk_list.new_chunk(chunk_id=1,
                             chunk_size=20,
                             data_type=torch.float,
                             compute_device=self.compute_device)
        assert chunk_list.size() == 2
        assert (chunk_list[1].get_status() == PSChunkStatus.RELEASED)
        chunk_list.delete_free_chunks()
        chunk_list.access_chunk(1, self.compute_device)

        chunk_list.visit()


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
