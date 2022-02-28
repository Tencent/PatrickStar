# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import unittest
from patrickstar.core.comm import CommInfo

import torch

from patrickstar.core import (
    ChunkTensorIndex,
    ChunkType,
    register_param,
    ParamType,
)


class TestAccess(unittest.TestCase):
    def setUp(self):
        pass

    def _check_order(self, chunk_tensor_index, chunk_id):
        start_offset_list = []
        for info in chunk_tensor_index.generate_tensor_info_in_order(chunk_id=chunk_id):
            start_offset = info.start_offset
            if len(start_offset_list) > 0:
                assert start_offset > start_offset_list[-1]
            start_offset_list.append(start_offset)

    def test_add_tensor(self):
        chunk_tensor_index = ChunkTensorIndex(1024)

        chunk_tensor_index.add_chunk(
            chunk_id=0,
            comm_info=CommInfo(chunk_type=ChunkType.PARAM_FP32, group_id=0, offset=0),
        )

        param_numel_list = [10, 20, 30, 20, 7, 2]
        param_list = []
        offset = 0
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(10))
            chunk_tensor_index.add_tensor(
                chunk_id=0,
                tensor_id=param_id,
                start_offset=offset,
                numel=numel,
                param=param,
            )
            offset += numel
            param_list.append(param)

        self._check_order(chunk_tensor_index, 0)

    def test_append_tensor(self):
        chunk_tensor_index = ChunkTensorIndex(20)
        param_numel_list = [10, 20, 30, 20, 7, 2]

        success_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(
                param, ParamType.CHUNK_BASED, torch.float, f"param_{param_id}"
            )
            is_success = chunk_tensor_index.try_insert_tensor(0, param)
            success_list.append(is_success)
        self.assertEqual(success_list, [True, False, False, False, True, True])

    def test_append_tensor_list(self):
        chunk_tensor_index = ChunkTensorIndex(20)
        param_numel_list = [7, 2]
        param_list = []

        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(
                param, ParamType.CHUNK_BASED, torch.float, f"param_{param_id}"
            )
            param_list.append(param)
        # 7， 2
        is_success = chunk_tensor_index.try_insert_tensor_list(0, param_list)
        self.assertTrue(is_success)

        # 7， 2， 6， 5
        param_numel_list = [6, 5]
        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(
                param, ParamType.CHUNK_BASED, torch.float, f"param_{param_id}"
            )
            param_list.append(param)

        is_success = chunk_tensor_index.try_insert_tensor_list(0, param_list)
        self.assertTrue(is_success)

        # 7， 2，(6), 5
        chunk_tensor_index.delete_tensor(0, param_list[0])
        param_numel_list = [8]
        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(
                param, ParamType.CHUNK_BASED, torch.float, f"param_{param_id}"
            )
            param_list.append(param)
        is_success = chunk_tensor_index.try_insert_tensor_list(0, param_list)
        self.assertFalse(is_success)

        # 7， 2，(6) 5
        param_numel_list = [7]
        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(
                param, ParamType.CHUNK_BASED, torch.float, f"param_{param_id}"
            )
            param_list.append(param)
        is_success = chunk_tensor_index.try_insert_tensor_list(0, param_list)
        self.assertFalse(is_success)

        # 7， 2，(6) 5
        param_numel_list = [1, 2, 3]
        param_list = []
        for param_id, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(
                param, ParamType.CHUNK_BASED, torch.float, f"param_{param_id}"
            )
            param_list.append(param)
        is_success = chunk_tensor_index.try_insert_tensor_list(0, param_list)
        self.assertTrue(is_success)

    def test_chunk_layout_consistency(self):
        r"""
        Check if the chunk layout of optimizer state are aligned to
        param fp16.
        """
        chunk_tensor_index = ChunkTensorIndex(20)

        param_numel_list = [10, 5]
        param_list = []

        for _, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float, f"param_{numel}")
            chunk_tensor_index.try_insert_tensor(0, param)
            param_list.append(param)

        param_numel_list = [6, 9]
        for _, numel in enumerate(param_numel_list):
            param = torch.nn.Parameter(torch.zeros(numel))
            register_param(param, ParamType.CHUNK_BASED, torch.float, f"param_{numel}")
            chunk_tensor_index.try_insert_tensor(1, param)
            param_list.append(param)

        # Now, we have 2 chunks, (10, 5) (6, 9)
        param_momentum = torch.nn.Parameter(torch.zeros(10))
        register_param(
            param_momentum, ParamType.CHUNK_BASED, torch.float, f"param_{numel}"
        )
        chunk_id = chunk_tensor_index.get_optimizer_state_chunk_id(
            param_list[0], ChunkType.MOMENTUM
        )
        self.assertTrue(chunk_id is None)

        chunk_tensor_index.register_optimizer_state_chunk_id(
            param_list[0], ChunkType.MOMENTUM, 3
        )
        chunk_id = chunk_tensor_index.get_optimizer_state_chunk_id(
            param_list[0], ChunkType.MOMENTUM
        )
        self.assertTrue(chunk_id == 3, f"chunk_id is {chunk_id} should be 3")


if __name__ == "__main__":
    unittest.main()
