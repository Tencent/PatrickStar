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

import torch

from common import distributed_test
from patrickstar.core import AccessType, ChunkTensorIndex
from patrickstar.core import register_param, ParamType


class TestChunkData(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40

    @distributed_test(world_size=[1])
    def test_allocate(self):
        self.compute_device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Statically construct chunk layout -> chunk_tensor_index
        chunk_tensor_index = ChunkTensorIndex(self.default_chunk_size)

        param1 = torch.nn.Parameter(torch.zeros(10))
        register_param(param1, ParamType.CHUNK_BASED, torch.float, "param1")
        chunk_tensor_index.add_tensor(
            chunk_id=0,
            tensor_id=param1.ps_attr.data_id(),
            start_offset=0,
            numel=param1.numel(),
            param=param1,
            access_type=AccessType.DATA,
        )

        self.assertTrue(
            chunk_tensor_index.tensor_id_to_chunk_id(param1.ps_attr.data_id()) == 0
        )
        self.assertTrue(chunk_tensor_index.get_chunk_id(param1, AccessType.DATA) == 0)

        param2 = torch.nn.Parameter(torch.zeros(15))
        register_param(param2, ParamType.CHUNK_BASED, torch.float, "param2")
        self.assertTrue(
            chunk_tensor_index.get_chunk_id(param2, AccessType.DATA) is None
        )
        ret = chunk_tensor_index.try_insert_tensor(0, param2, AccessType.DATA)
        self.assertTrue(ret)
        tensor_info = chunk_tensor_index.get_tensor_info(param2.ps_attr.data_id())
        self.assertTrue(tensor_info.start_offset == 10)

        param3 = torch.nn.Parameter(torch.zeros(5))
        register_param(param3, ParamType.CHUNK_BASED, torch.float, "param3")
        ret = chunk_tensor_index.try_insert_tensor(0, param3, AccessType.DATA)
        tensor_info = chunk_tensor_index.get_tensor_info(param3.ps_attr.data_id())
        self.assertTrue(tensor_info.start_offset == 25)

        param4 = torch.nn.Parameter(torch.zeros(100))
        register_param(param4, ParamType.CHUNK_BASED, torch.float, "param4")
        ret = chunk_tensor_index.try_insert_tensor(0, param4, AccessType.DATA)
        self.assertFalse(ret)
        # chunk_tensor_index.delete_tensor(11)

        param5 = torch.nn.Parameter(torch.zeros(13))
        register_param(param5, ParamType.CHUNK_BASED, torch.float, "param5")
        ret = chunk_tensor_index.try_insert_tensor(1, param5, AccessType.DATA)
        tensor_info = chunk_tensor_index.get_tensor_info(param5.ps_attr.data_id())
        self.assertTrue(tensor_info.start_offset == 0)

        ret = chunk_tensor_index.try_insert_tensor(1, param5, AccessType.DATA)
        tensor_info = chunk_tensor_index.get_tensor_info(param5.ps_attr.data_id())
        self.assertTrue(tensor_info.start_offset == 0)

        param6 = torch.nn.Parameter(torch.zeros(1000))
        register_param(param6, ParamType.CHUNK_BASED, torch.float, "param6")
        ret = chunk_tensor_index.try_insert_tensor(1, param6, AccessType.DATA)
        self.assertFalse(ret)


if __name__ == "__main__":

    unittest.main()
