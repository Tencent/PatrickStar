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
from client import HybridPSClient, ChunkList, PSTensorStatus, AccessType, ChunkTensorIndex, PSChunkStatus
import logging
import torch
from manager import HybridPSManager
from utils import see_memory_usage
from client import register_param


class TestClientAccess(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 20
        self.client = HybridPSClient(
            gpu_index=0, default_chunk_size=self.default_chunk_size)
        self.manager = HybridPSManager()
        self.manager.reset([300], [1024])
        self.compute_device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.prepared_chunk_schema()
        logging.info('SetUp finished')

    def prepared_chunk_schema(self):
        self.chunk_tensor_index = self.client.chunk_tensor_index
        # 构建静态chunk layout -> chunk_tensor_index
        self.param1 = torch.nn.Parameter(torch.zeros(10))
        register_param(self.param1, 'param1')
        self.chunk_tensor_index.add_tensor(0, self.param1.ps_attr.data_id(), 0,
                                           self.param1.numel(), self.param1,
                                           AccessType.DATA)

        self.param2 = torch.nn.Parameter(torch.zeros(15))
        register_param(self.param2, 'param2')
        self.chunk_tensor_index.add_tensor(0,
                                           self.param2.ps_attr.data_id(), 20,
                                           self.param2.numel(), self.param2,
                                           AccessType.DATA)

        self.param3 = torch.nn.Parameter(torch.zeros(5))
        register_param(self.param3, 'param3')
        self.chunk_tensor_index.add_tensor(0,
                                           self.param3.ps_attr.data_id(), 15,
                                           self.param3.numel(), self.param3,
                                           AccessType.DATA)

        self.param4 = torch.nn.Parameter(torch.zeros(7))
        register_param(self.param4, 'param4')
        self.chunk_tensor_index.add_tensor(0,
                                           self.param4.ps_attr.data_id(), 35,
                                           self.param4.numel(), self.param4,
                                           AccessType.DATA)

        # chunk_tensor_index.delete_tensor(11)

        self.param5 = torch.nn.Parameter(torch.zeros(13))
        register_param(self.param5, 'param5')
        self.chunk_tensor_index.add_tensor(1, self.param5.ps_attr.data_id(), 7,
                                           self.param5.numel(), self.param5,
                                           AccessType.DATA)

        self.param6 = torch.nn.Parameter(torch.zeros(3))
        register_param(self.param6, 'param6')
        self.chunk_tensor_index.add_tensor(1, self.param6.ps_attr.data_id(), 2,
                                           self.param6.numel(), self.param6,
                                           AccessType.DATA)

        # 初始化chunk
        self.client.chunk_list.new_chunk(0, 50, torch.float)
        self.client.chunk_list.new_chunk(1, 20, torch.float)
        logging.info('prepared_chunk_schema finished')

    def _access_single_tensor(self, param):
        self.client.access_data(param, self.compute_device)
        assert param.ps_attr.access_tensor(
            AccessType.DATA).device == self.compute_device

    def test_access_multiple_tensor(self):
        self._access_single_tensor(self.param1)
        self._access_single_tensor(self.param2)
        self._access_single_tensor(self.param3)
        self._access_single_tensor(self.param4)
        self._access_single_tensor(self.param5)
        self._access_single_tensor(self.param6)

        assert self.client.chunk_list[0].get_status() == PSChunkStatus.COMPUTE
        assert self.client.chunk_list[1].get_status() == PSChunkStatus.COMPUTE

    def _release_param(self, param, status: PSTensorStatus):
        """
        测试单个param的申请和释放
        """
        self.client.access_data(param, self.compute_device)
        assert self.client.chunk_list[0].get_status() == PSChunkStatus.COMPUTE
        self.client.release_data(param, status)
        assert param.ps_attr.get_status(
            AccessType.DATA
        ) == status, f"param1 status is {param.ps_attr.get_status(AccessType.DATA)}"

    def test_release_tensors(self):
        self._release_param(self.param1, PSTensorStatus.FREE)
        assert self.client.chunk_list[0].get_status() == PSChunkStatus.RELEASED
        self._release_param(self.param2, PSTensorStatus.HOLD)
        assert self.client.chunk_list[0].get_status() == PSChunkStatus.HOLD

    def test_room_making(self):
        # total 300 B
        # chunk 0 (200B), chunk 1 (80B)
        self._access_single_tensor(self.param1)
        self._access_single_tensor(self.param2)
        self._access_single_tensor(self.param3)
        self._access_single_tensor(self.param4)
        self._access_single_tensor(self.param5)
        self._access_single_tensor(self.param6)
        self._release_param(self.param5, PSTensorStatus.FREE)
        self._release_param(self.param6, PSTensorStatus.HOLD)
        assert self.client.chunk_list[1].get_status() == PSChunkStatus.HOLD
        assert self.client.chunk_list[1].get_device() == self.compute_device

        # chunk 2 再申请 100B，这是需要从GPU中移除 chunk 1
        self.param7 = torch.nn.Parameter(torch.zeros(20))
        register_param(self.param7, 'param7')
        self.chunk_tensor_index.add_tensor(2, self.param7.ps_attr.data_id(), 0,
                                           self.param7.numel(), self.param7,
                                           AccessType.DATA)
        self.client.chunk_list.new_chunk(2, 25, torch.float)

        # *检查显存是否真正释放, 1024 B
        see_memory_usage(
            f"before release a chunk of size {self.client.chunk_list[1].get_size()} B and allocate a chunk of size {self.client.chunk_list[2].get_size()} B",
            True, "B")

        self.client.access_data(self.param7, self.compute_device)

        # *检查显存是否真正释放, 1024 B
        see_memory_usage(
            f"after release a chunk of size {self.client.chunk_list[1].get_size()} B and allocate a chunk of size {self.client.chunk_list[2].get_size()} B",
            True, "B")

        assert self.client.chunk_list[1].get_device() == torch.device('cpu')
        assert self.client.chunk_list[2].get_device() == self.compute_device
        # self.client.chunk_list.visit()

    def test_room_releasing(self):
        # total 300 B
        # chunk 0 (200B), chunk 1 (80B)
        self._access_single_tensor(self.param1)
        self._access_single_tensor(self.param2)
        self._access_single_tensor(self.param3)
        self._access_single_tensor(self.param4)
        self._access_single_tensor(self.param5)
        self._access_single_tensor(self.param6)

        # 将chunk 1释放
        self._release_param(self.param5, PSTensorStatus.FREE)
        assert self.client.chunk_list[1].get_status() == PSChunkStatus.COMPUTE

        # *检查显存是否真正释放, 1024 B
        see_memory_usage(
            f"before release a chunk of size {self.client.chunk_list[1].get_size()} B",
            True, "B")
        self._release_param(self.param6, PSTensorStatus.FREE)
        print(self.client.chunk_list[1].get_status())
        assert self.client.chunk_list[1].get_status() == PSChunkStatus.RELEASED
        assert self.client.chunk_list[1].get_device() == None

        # *检查显存是否真正释放, 512 B
        torch.cuda.empty_cache()
        see_memory_usage(
            f"after release a chunk of size {self.client.chunk_list[1].get_size()} B",
            True, "B")

        # self.client.chunk_list.visit()

        # chunk 2 再申请 100B，这是需要从GPU删除1
        self.param7 = torch.nn.Parameter(torch.zeros(20))
        register_param(self.param7, 'param7')
        self.chunk_tensor_index.add_tensor(2, self.param7.ps_attr.data_id(), 0,
                                           self.param7.numel(), self.param7,
                                           AccessType.DATA)
        self.client.chunk_list.new_chunk(2, 25, torch.float)
        self.client.access_data(self.param7, self.compute_device)
        assert self.client.chunk_list[1].get_device() == None
        assert self.client.chunk_list[2].get_device() == self.compute_device
        # self.client.chunk_list.visit()

        # 释放掉chunk 2, 再访问chunk 1
        self._release_param(self.param7, PSTensorStatus.HOLD)
        self._access_single_tensor(self.param5)
        assert self.client.chunk_list[2].get_device() == torch.device('cpu')
        self.client.chunk_list.visit()


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
