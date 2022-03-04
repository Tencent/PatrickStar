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
        metronome.set_warmup(True)
        policy = LatestAccessChunkEvictionPolicy(metronome)

        # trace chunk access
        policy.trace_access(0, dev)
        metronome.tiktac()
        policy.trace_access(1, dev)
        print(policy.chunk_access_dict)

        # Finish warmup
        metronome.set_warmup(False)
        metronome.reset()

        # Test eviction strategy
        ret_list = policy.derive_eviction_list(id_to_chunk_list, 10, dev)
        self.assertTrue(ret_list == [0])

        metronome.tiktac()
        ret_list = policy.derive_eviction_list(id_to_chunk_list, 10, dev)
        self.assertTrue(ret_list == [1])


if __name__ == "__main__":
    unittest.main()
