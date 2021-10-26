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
from patrickstar.core import ChunkList, ChunkState, ChunkType


class TestChunkData(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_add_chunk(self):
        chunk_list = ChunkList(0)
        assert chunk_list.size() == 0

        chunk_list.new_chunk(
            chunk_id=0,
            chunk_size=20,
            data_type=torch.float,
            is_dummy=False,
            chunk_type=ChunkType.PARAM_FP32,
        )

        assert chunk_list.size() == 1
        assert chunk_list[0].get_state() == ChunkState.RELEASED

    @distributed_test(world_size=[1], use_fake_dist=True)
    def test_new_chunk(self):
        compute_device = (
            torch.device(f"cuda:{torch.cuda.current_device()}")
            if torch.cuda.is_available()
            else torch.device("cpu:0")
        )
        chunk_list = ChunkList(0)

        new_chunk_id = 123
        chunk_list.new_chunk(
            chunk_id=new_chunk_id,
            chunk_size=20,
            data_type=torch.float,
            is_dummy=False,
            chunk_type=ChunkType.PARAM_FP32,
        )
        chunk_list.access_chunk(new_chunk_id, compute_device)

        assert chunk_list[new_chunk_id].get_state() == ChunkState.FREE

        self.assertEqual(
            chunk_list.last_chunk_id(ChunkType.PARAM_FP32),
            new_chunk_id,
            "check last_chunk_id",
        )

        chunk_list.new_chunk(
            chunk_id=1,
            chunk_size=20,
            data_type=torch.float,
            is_dummy=False,
            chunk_type=ChunkType.PARAM_FP32,
        )

        self.assertEqual(chunk_list.size(), 2)

        self.assertEqual(
            chunk_list.last_chunk_id(ChunkType.PARAM_FP32), 1, "check last_chunk_id"
        )


if __name__ == "__main__":

    unittest.main()
