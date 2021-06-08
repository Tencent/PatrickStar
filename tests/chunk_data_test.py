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
from client import PatrickStarClient, ChunkList, PSTensorStatus, AccessType, ChunkTensorIndex, Chunk
import logging
import torch
from manager import PatrickStarManager
from client import PSChunkStatus


class TestChunkData(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40
        self.client = PatrickStarClient(
            rank=0, default_chunk_size=self.default_chunk_size)
        self.manager = PatrickStarManager()
        self.manager.init([32, 32], [1024])
        self.compute_device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    def test_allocate(self):
        # 构建静态chunk layout -> chunk_tensor_index
        chunk_tensor_index = ChunkTensorIndex()
        self.client.chunk_tensor_index = chunk_tensor_index

        param1 = torch.nn.Parameter(torch.zeros(10))
        param1.ps_data_id = 10
        param1.ps_shape = param1.shape
        chunk_tensor_index.add_tensor(0, param1.ps_data_id, 0, param1.numel(),
                                      param1, AccessType.DATA)

        param2 = torch.nn.Parameter(torch.zeros(15))
        param2.ps_data_id = 11
        param2.ps_shape = param2.shape
        chunk_tensor_index.add_tensor(0, param2.ps_data_id, 20, param2.numel(),
                                      param2, AccessType.DATA)

        param3 = torch.nn.Parameter(torch.zeros(5))
        param3.ps_data_id = 12
        param3.ps_shape = param3.shape
        chunk_tensor_index.add_tensor(0, param3.ps_data_id, 15, param3.numel(),
                                      param3, AccessType.DATA)

        param4 = torch.nn.Parameter(torch.zeros(7))
        param4.ps_data_id = 13
        param4.ps_shape = param4.shape
        chunk_tensor_index.add_tensor(0, param4.ps_data_id, 35, param4.numel(),
                                      param4, AccessType.DATA)

        # chunk_tensor_index.delete_tensor(11)

        param5 = torch.nn.Parameter(torch.zeros(13))
        param5.ps_data_id = 14
        param5.ps_shape = param5.shape
        chunk_tensor_index.add_tensor(1, param5.ps_data_id, 7, param5.numel(),
                                      param5, AccessType.DATA)

        param6 = torch.nn.Parameter(torch.zeros(3))
        param6.ps_data_id = 15
        param6.ps_shape = param6.shape
        chunk_tensor_index.add_tensor(1, param6.ps_data_id, 2, param6.numel(),
                                      param6, AccessType.DATA)

        # chunk_tensor_index.delete_tensor(14)
        # assert (chunk_tensor_index.tensor_id_to_chunk_id(14) is None)

        # chunk_tensor_index.delete_tensor(15)
        # assert (chunk_tensor_index.tensor_id_to_chunk_id(14) is None)

        # 初始化chunk
        chunk1 = Chunk(100, torch.float, 0, self.compute_device)
        chunk2 = Chunk(100, torch.float, 1, self.compute_device)
        assert chunk1.get_device() is None

        chunk1.allocate_payload(self.compute_device)
        print(chunk1.get_device())
        assert chunk1.get_device() == self.compute_device

        # 测试param1访问
        chunk1.access_param(param1, AccessType.DATA, chunk_tensor_index)
        assert param1.ps_data_tensor.numel() == 10
        assert param1.ps_data_tensor.device == self.compute_device
        # print(param1.ps_data_tensor)

        # 测试param1移动
        chunk1.move(chunk_tensor_index, torch.device('cpu'))
        assert param1.ps_data_tensor.device == torch.device('cpu')
        print(param1.ps_data_tensor)

        # chunk1.visit(chunk_tensor_index)
        assert chunk1.get_status() == PSChunkStatus.FREE

        chunk2.visit(chunk_tensor_index)


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
