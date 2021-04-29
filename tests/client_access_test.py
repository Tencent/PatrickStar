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


class TestAccess(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 20
        self.client = HybridPSClient(
            gpu_index=0, default_chunk_size=self.default_chunk_size)
        self.manager = HybridPSManager()
        self.manager.init([32, 32], [1024])
        self.compute_device = torch.device(
            'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        logging.info('SetUp finished')

    def test_register(self):
        self.manager.reset([40 * 4], [256 * 4])
        param1 = torch.nn.Parameter(torch.zeros(23, dtype=torch.float))
        assert param1.requires_grad is True
        self.client.register_param(param1)
        assert param1.data_status == PSTensorStatus.HOLD
        assert param1.grad_status == PSTensorStatus.UNINIT

    def test_access(self):
        self.manager.reset([40 * 4], [256 * 4])
        param1 = torch.nn.Parameter(torch.zeros(10, dtype=torch.float))

        assert param1.requires_grad is True

        logging.info(f'access param1 data')
        self.client.register_param(param1)
        # id = 0
        self.client.access_data(param1, self.compute_device)
        assert param1.data_status == PSTensorStatus.COMPUTE
        assert param1.grad_status == PSTensorStatus.UNINIT
        assert self.client.get_chunk_id(param1, AccessType.DATA) == 0

        logging.info(f'access param1 grad')
        # id = 1
        self.client.access_grad(param1, self.compute_device)
        assert param1.data_status == PSTensorStatus.COMPUTE
        assert param1.grad_status == PSTensorStatus.COMPUTE
        assert self.client.get_chunk_id(param1, AccessType.GRAD) == 0

        logging.info(f'access param2 data')
        param2 = torch.nn.Parameter(torch.zeros(10, dtype=torch.float))
        self.client.register_param(param2)

        logging.info(f'release param1 grad')
        self.client.release_grad(param1, PSTensorStatus.FREE)

        logging.info(f'release param2 grad')
        # id = 3
        self.client.access_grad(param2, self.compute_device)
        assert self.client.get_chunk_id(param2, AccessType.GRAD) == 0

        self.client.visit()
        param2_numel = param2.ps_shape.numel()
        see_memory_usage(f"====before access a chunk of numel {param2_numel}",
                         force=True,
                         scale_name="B")

        a = {0: torch.zeros(self.default_chunk_size)}
        b = a[0].narrow(0, 0, 1)
        a[0] = a[0].to(self.compute_device)
        b = a[0].narrow(0, 0, 1)
        see_memory_usage(f"====allocate a torch tensor on GPU",
                         force=True,
                         scale_name="B")

        del a[0]
        del b
        see_memory_usage(f"====release the torch tensor from GPU",
                         force=True,
                         scale_name="B")

        # 删除chunk内存了
        self.client.access_data(param2, self.compute_device)
        # self.client.visit()
        see_memory_usage(f"====before release a chunk of numel {param2_numel}",
                         force=True,
                         scale_name="B")
        self.client.release_data(param2, PSTensorStatus.FREE)
        see_memory_usage(f"====after release a chunk of numel {param2_numel}",
                         force=True,
                         scale_name="B")
        self.client.visit()

    def test_room_making(self):
        self.manager.reset([40 * 4], [256 * 4])
        param1 = torch.nn.Parameter(torch.zeros(10, dtype=torch.float))
        param2 = torch.nn.Parameter(torch.zeros(10, dtype=torch.float))
        param3 = torch.nn.Parameter(torch.zeros(5, dtype=torch.float))

        # 先把显存填满
        self.client.register_param(param1)
        self.client.access_data(param1, self.compute_device)
        self.client.access_grad(param1, self.compute_device)
        self.client.register_param(param2)
        self.client.access_data(param2, self.compute_device)
        self.client.visit()
        self.client.access_grad(param2, self.compute_device)

        # 把别人挤出去
        self.client.visit()
        self.client.register_param(param3)
        logging.info('==== after register param ===')
        self.client.visit()
        self.client.release_data(param2)
        self.client.release_grad(param2)
        self.client.access_data(param3, self.compute_device)
        self.client.access_grad(param3, self.compute_device)
        logging.info('==== after access_data param ===')
        self.client.visit()

        assert param2.ps_data_tensor.device == torch.device('cpu')
        assert param2.ps_grad_tensor.device == torch.device('cpu')
        assert param3.ps_data_tensor.device == self.compute_device
        assert param3.ps_grad_tensor.device == self.compute_device


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
