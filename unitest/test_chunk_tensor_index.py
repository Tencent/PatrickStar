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

import torch

from patrickstar.core import ChunkTensorIndex, ChunkListType, AccessType, register_param, ParamType


class TestAccess(unittest.TestCase):
    def setUp(self):
        pass

    def _check_order(self, chunk_tensor_index, chunk_id):
        start_offset_list = []
        for info in chunk_tensor_index.generate_tensor_info_in_order(
                chunk_id=chunk_id):
            start_offset = info.start_offset
            if len(start_offset_list) > 0:
                assert (start_offset > start_offset_list[-1])
            start_offset_list.append(start_offset)

    def test_add_tensor(self):
        chunk_tensor_index = ChunkTensorIndex(1024)

        chunk_tensor_index.add_chunk(chunk_id=0,
                                     chunk_size=1024,
                                     data_type=torch.float,
                                     comm_group_id=0,
                                     comm_group_offset=0,
                                     list_type=ChunkListType.PARAM_FP32)

        param_numel_list = [10, 20, 30, 20, 7, 2]
        param_list = []
        offset = 0
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(10))
            chunk_tensor_index.add_tensor(chunk_id=0,
                                          tensor_id=param_id,
                                          start_offset=offset,
                                          numel=numel,
                                          param=param,
                                          access_type=AccessType.DATA)
            offset += numel
            param_list.append(param)

        self._check_order(chunk_tensor_index, 0)

    def test_append_tensor(self):
        chunk_tensor_index = ChunkTensorIndex(20)
        param_numel_list = [10, 20, 30, 20, 7, 2]
        param_list = []

        success_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{param_id}")
            is_success = chunk_tensor_index.try_insert_tensor(
                0, param, torch.float, AccessType.DATA)
            success_list.append(is_success)
        self.assertEqual(success_list, [True, False, False, False, True, True])

    def test_append_tensor_list(self):
        chunk_tensor_index = ChunkTensorIndex(20)
        param_numel_list = [7, 2]
        param_list = []

        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{param_id}")
            param_list.append(param)
        # 7， 2
        is_success = chunk_tensor_index.try_insert_tensor_list(
            0, param_list, torch.float, AccessType.DATA)
        self.assertTrue(is_success)

        # 7， 2， 6， 5
        param_numel_list = [6, 5]
        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{param_id}")
            param_list.append(param)

        is_success = chunk_tensor_index.try_insert_tensor_list(
            0, param_list, torch.float, AccessType.DATA)
        self.assertTrue(is_success)

        # 7， 2，(6), 5
        chunk_tensor_index.delete_tensor(0, param_list[0], AccessType.DATA)
        param_numel_list = [8]
        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{param_id}")
            param_list.append(param)
        is_success = chunk_tensor_index.try_insert_tensor_list(
            0, param_list, torch.float, AccessType.DATA)
        self.assertFalse(is_success)

        # 7， 2，(6) 5
        param_numel_list = [7]
        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{param_id}")
            param_list.append(param)
        is_success = chunk_tensor_index.try_insert_tensor_list(
            0, param_list, torch.float, AccessType.DATA)
        self.assertFalse(is_success)

        # 7， 2，(6) 5
        param_numel_list = [1, 2, 3]
        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{param_id}")
            param_list.append(param)
        is_success = chunk_tensor_index.try_insert_tensor_list(
            0, param_list, torch.float, AccessType.DATA)
        self.assertTrue(is_success)

    def test_chunk_layout_consistency(self):
        """
        检查OS的chunk layout是否和param fp16对齐
        """
        chunk_tensor_index = ChunkTensorIndex(20)

        param_numel_list = [10, 5]
        param_list = []

        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{numel}")
            is_success = chunk_tensor_index.try_insert_tensor(
                0, param, torch.float, AccessType.DATA)
            param_list.append(param)

        param_numel_list = [6, 9]
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float,
                           f"param_{numel}")
            is_success = chunk_tensor_index.try_insert_tensor(
                1, param, torch.float, AccessType.DATA)
            param_list.append(param)

        # Now, we have 2 chunks, (10, 5) (6, 9)
        param_momentum = torch.nn.Parameter(torch.zeros(10))
        register_param(param_momentum, ParamType.CHUNK_BASED, torch.float,
                       f"param_{numel}")
        chunk_id = chunk_tensor_index.get_optimizer_state_chunk_id(
            param_list[0], AccessType.DATA, ChunkListType.MOMENTUM)
        self.assertTrue(chunk_id is None)

        chunk_tensor_index.register_optimizer_state_chunk_id(
            param_list[0], AccessType.DATA, ChunkListType.MOMENTUM, 3)
        chunk_id = chunk_tensor_index.get_optimizer_state_chunk_id(
            param_list[0], AccessType.DATA, ChunkListType.MOMENTUM)
        self.assertTrue(chunk_id == 3, f"chunk_id is {chunk_id} should be 3")


if __name__ == "__main__":
    unittest.main()
