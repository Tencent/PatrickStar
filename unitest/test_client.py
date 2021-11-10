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

import logging
import unittest

import torch

from common import distributed_test
from patrickstar import RuntimeMemTracer
from patrickstar.core import PatrickStarClient, AccessType, register_param, ChunkType
from patrickstar.core.parameter import ParamType


class TestClientAccess(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40
        logging.info("SetUp finished")

    @distributed_test(world_size=[1])
    def test_append_ps_tensor(self):
        RuntimeMemTracer(0)
        self.client = PatrickStarClient(
            rank=0, default_chunk_size=self.default_chunk_size
        )

        self.compute_device = torch.device("cpu:0")

        param_size_list = [10, 11, 12, 13]

        param_list = []
        param_payload_ref_list = []
        for idx, psize in enumerate(param_size_list):
            param = torch.nn.Parameter(torch.rand(psize))
            param_list.append(param)
            param_payload_ref_list.append(param.data.clone())

            register_param(param, ParamType.CHUNK_BASED, torch.float, f"param_{idx}")
            self.client.append_tensor(
                [param], torch.float, AccessType.DATA, ChunkType.PARAM_FP32
            )

            real_payload = self.client.access_data(param, torch.device("cpu:0"))
            real_payload.copy_(param.data)
            self.client.release_data(param)
            self.assertTrue(param.data.numel() == 0)

        self.client.display_chunk_info()
        for param, payload_ref in zip(param_list, param_payload_ref_list):
            real_payload = self.client.access_data(param, torch.device("cpu:0"))
            self.assertEqual(torch.max(real_payload - payload_ref), 0)
            self.client.release_data(param)

    @distributed_test(world_size=[1])
    def test_append_torch_tensor(self):
        self.client = PatrickStarClient(
            rank=0, default_chunk_size=self.default_chunk_size
        )

        self.compute_device = torch.device("cpu:0")

        param_size_list = [10, 11, 12, 13]

        param_list = []
        param_payload_ref_list = []
        for idx, psize in enumerate(param_size_list):
            param = torch.nn.Parameter(torch.rand(psize))
            param_list.append(param)
            register_param(param, ParamType.TORCH_BASED, torch.float, f"param_{idx}")
            param_payload_ref_list.append(param.data.clone())
            self.client.append_tensor(
                [param], torch.float, AccessType.DATA, ChunkType.PARAM_FP32
            )

            real_payload = self.client.access_data(param, torch.device("cpu:0"))
            real_payload.copy_(param.data)
            self.client.release_data(param)

        self.client.display_chunk_info()
        for param, payload_ref in zip(param_list, param_payload_ref_list):
            real_payload = self.client.access_data(param, torch.device("cpu:0"))
            self.assertEqual(torch.max(real_payload - payload_ref), 0)
            self.client.release_data(param)


if __name__ == "__main__":

    unittest.main()
