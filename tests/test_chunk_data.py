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
from patrickstar.core import PatrickStarClient, ChunkList, PSTensorStatus, AccessType, ChunkTensorIndex, Chunk
import logging
import torch
from patrickstar.core import PSChunkStatus, register_param, ParamType
from common import distributed_test
from patrickstar import PatrickStarManager


class TestChunkData(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40

    @distributed_test(world_size=[1])
    def test_allocate(self):
        self.compute_device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # 构建静态chunk layout -> chunk_tensor_index
        chunk_tensor_index = ChunkTensorIndex()

        param1 = torch.nn.Parameter(torch.zeros(10))
        register_param(param1, ParamType.CHUNK_BASED, torch.float, "param1")
        chunk_tensor_index.add_tensor(chunk_id=0,
                                      tensor_id=param1.ps_attr.data_id(),
                                      start_offset=0,
                                      numel=param1.numel(),
                                      param=param1,
                                      access_type=AccessType.DATA)

        self.assertTrue(
            chunk_tensor_index.tensor_id_to_chunk_id(param1.ps_attr.data_id())
            == 0)
        self.assertTrue(
            chunk_tensor_index.get_chunk_id(param1, AccessType.DATA) == 0)

        param2 = torch.nn.Parameter(torch.zeros(15))
        register_param(param2, ParamType.CHUNK_BASED, torch.float, "param2")
        self.assertTrue(
            chunk_tensor_index.get_chunk_id(param2, AccessType.DATA) is None)
        chunk_tensor_index.try_insert_tensor(0, param2, torch.float,
                                             AccessType.DATA)
        tensor_info = chunk_tensor_index.get_tensor_info(
            param2.ps_attr.data_id())
        self.assertTrue(tensor_info.offset)
        # chunk_tensor_index.add_tensor(0, param2.ps_attr.data_id(), 20, param2.numel(),
        #                               param2, AccessType.DATA)

        param3 = torch.nn.Parameter(torch.zeros(5))
        register_param(param3, ParamType.CHUNK_BASED, torch.float, "param3")
        chunk_tensor_index.add_tensor(0, param3.ps_attr.data_id(), 15,
                                      param3.numel(), param3, AccessType.DATA)

        param4 = torch.nn.Parameter(torch.zeros(7))
        register_param(param4, ParamType.CHUNK_BASED, torch.float, "param4")
        chunk_tensor_index.add_tensor(0, param4.ps_attr.data_id(), 35,
                                      param4.numel(), param4, AccessType.DATA)

        # chunk_tensor_index.delete_tensor(11)

        param5 = torch.nn.Parameter(torch.zeros(13))
        register_param(param5, ParamType.CHUNK_BASED, torch.float, "param5")
        chunk_tensor_index.add_tensor(1, param5.ps_attr.data_id(), 7,
                                      param5.numel(), param5, AccessType.DATA)

        param6 = torch.nn.Parameter(torch.zeros(3))
        register_param(param6, ParamType.CHUNK_BASED, torch.float, "param6")
        chunk_tensor_index.add_tensor(1, param6.ps_attr.data_id(), 2,
                                      param6.numel(), param6, AccessType.DATA)

        # chunk_tensor_index.visit_chunk()


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
