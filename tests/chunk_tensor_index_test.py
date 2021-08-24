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
from patrickstar.core import ChunkTensorIndex, ChunkListType, AccessType
import logging
import torch


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
        chunk_tensor_index = ChunkTensorIndex()

        chunk_tensor_index.add_chunk(chunk_id=0,
                                     chunk_size=1000,
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


if __name__ == "__main__":
    unittest.main()
