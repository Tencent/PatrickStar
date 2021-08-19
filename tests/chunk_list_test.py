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
from patrickstar.core import PatrickStarClient, ChunkList, PSTensorStatus, AccessType, ChunkTensorIndex, Chunk, PSChunkStatus, ChunkListType
import logging
import torch
from common import distributed_test
from patrickstar.deepspeed_helper.global_vars import set_global_variables, get_args


class TestChunkData(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_add_chunk(self):
        args = get_args()
        print('local rank', args.local_rank)
        default_chunk_size = 40

        compute_device = torch.device(
            f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available(
            ) else torch.device('cpu:0')

        chunk_list = ChunkList()
        assert chunk_list.size() == 0

        chunk_list.new_chunk(chunk_id=0,
                             chunk_size=20,
                             data_type=torch.float,
                             is_dummy=False,
                             chunk_type=ChunkListType.PARAM_FP32)

        assert chunk_list.size() == 1
        assert (chunk_list[0].get_status() == PSChunkStatus.RELEASED)

    @distributed_test(world_size=[1])
    def test_new_chunk(self):
        compute_device = torch.device(
            f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available(
            ) else torch.device('cpu:0')
        chunk_list = ChunkList()

        new_chunk_id = 123
        chunk_list.new_chunk(chunk_id=new_chunk_id,
                             chunk_size=20,
                             data_type=torch.float,
                             is_dummy=False,
                             chunk_type=ChunkListType.PARAM_FP32)
        chunk_list.access_chunk(new_chunk_id, compute_device)

        assert (chunk_list[new_chunk_id].get_status() == PSChunkStatus.FREE)

        self.assertEqual(chunk_list.last_chunk_id(ChunkListType.PARAM_FP32),
                         new_chunk_id, "check last_chunk_id")

        chunk_list.new_chunk(chunk_id=1,
                             chunk_size=20,
                             data_type=torch.float,
                             is_dummy=False,
                             chunk_type=ChunkListType.PARAM_FP32)

        self.assertEqual(chunk_list.size(), 2)

        self.assertEqual(chunk_list.last_chunk_id(ChunkListType.PARAM_FP32), 1,
                         "check last_chunk_id")


if __name__ == "__main__":
    set_global_variables()
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
