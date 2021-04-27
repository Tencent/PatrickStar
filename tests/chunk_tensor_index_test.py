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
from client import HybridPSClient, ChunkList, PSTensorStatus, AccessType, ChunkTensorIndex
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

    def _check_order(self, chunk_tensor_index, chunk_id):
        start_offset_list = []
        for info in chunk_tensor_index.generate_tensor_info_in_order(
                chunk_id=chunk_id):
            start_offset = info.start_offset
            if len(start_offset_list) > 0:
                assert (start_offset > start_offset_list[-1])
            start_offset_list.append(start_offset)
            info.showme()

    def test_add_tensor(self):
        chunk_tensor_index = ChunkTensorIndex()
        param1 = torch.nn.Parameter(torch.zeros(10))
        # self.client.register_param(param1)
        chunk_tensor_index.add_tensor(0, 10, 0, param1.numel(), param1,
                                      AccessType.DATA)

        param2 = torch.nn.Parameter(torch.zeros(15))
        # self.client.register_param(param2)
        chunk_tensor_index.add_tensor(0, 11, 20, param2.numel(), param2,
                                      AccessType.DATA)

        param3 = torch.nn.Parameter(torch.zeros(5))
        # self.client.register_param(param2)
        chunk_tensor_index.add_tensor(0, 12, 15, param3.numel(), param3,
                                      AccessType.DATA)

        param4 = torch.nn.Parameter(torch.zeros(7))
        # self.client.register_param(param2)
        chunk_tensor_index.add_tensor(0, 13, 35, param3.numel(), param4,
                                      AccessType.DATA)

        chunk_tensor_index.delete_tensor(11)

        param5 = torch.nn.Parameter(torch.zeros(13))
        chunk_tensor_index.add_tensor(1, 14, 7, param5.numel(), param5,
                                      AccessType.DATA)

        param6 = torch.nn.Parameter(torch.zeros(3))
        chunk_tensor_index.add_tensor(1, 15, 2, param6.numel(), param6,
                                      AccessType.DATA)

        chunk_tensor_index.delete_tensor(14)
        assert (chunk_tensor_index.tensor_id_to_chunk_id(14) is None)

        chunk_tensor_index.delete_tensor(15)
        assert (chunk_tensor_index.tensor_id_to_chunk_id(14) is None)

        self._check_order(chunk_tensor_index, 0)
        self._check_order(chunk_tensor_index, 1)


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
