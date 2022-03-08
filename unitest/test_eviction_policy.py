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
from patrickstar.core.eviction_policy import LatestAccessChunkEvictionPolicy
from patrickstar.core.chunk import Chunk
from patrickstar.core.memtracer import RuntimeMemTracer


class TestEvictionPolicy(unittest.TestCase):
    def setUp(self):
        pass

    def test_chunk_eviction(self):
        id_to_chunk_list = {}
        dev = torch.device("cpu:0")
        mem_tracer = RuntimeMemTracer(
            local_rank=0, config={"use_async_mem_monitor": True}
        )
        id_to_chunk_list[0] = Chunk(10, torch.float, 0, mem_tracer, None, 0, False)
        id_to_chunk_list[0].allocate_payload(dev)
        id_to_chunk_list[1] = Chunk(10, torch.float, 1, mem_tracer, None, 0, False)
        id_to_chunk_list[1].allocate_payload(dev)
        metronome = mem_tracer.metronome
        metronome.is_warmup = True
        policy = LatestAccessChunkEvictionPolicy(metronome)

        # trace chunk access
        policy.trace_access(0, dev)
        metronome.tiktac()
        policy.trace_access(1, dev)
        print(policy.chunk_access_dict)

        # Finish warmup
        metronome.is_warmup = False
        metronome.reset()

        # Test eviction strategy
        ret_list = policy.derive_eviction_list(id_to_chunk_list, 10, dev)
        self.assertTrue(ret_list == [0])

        metronome.tiktac()
        ret_list = policy.derive_eviction_list(id_to_chunk_list, 10, dev)
        self.assertTrue(ret_list == [1])


if __name__ == "__main__":
    unittest.main()
