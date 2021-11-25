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

from patrickstar.core.memory_cache import MemoryCache
from patrickstar.core.memtracer import RuntimeMemTracer


class TestMemoryCache(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40

    def test_case1(self):
        self.compute_device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        memtracer = RuntimeMemTracer()
        memory_cache = MemoryCache(2, memtracer)

        payload1 = memory_cache.pop_or_allocate(self.compute_device, 10, torch.float)
        payload1_addr = payload1.data_ptr()
        memory_cache.push(payload1)
        payload2 = memory_cache.pop_or_allocate(self.compute_device, 10, torch.float)
        self.assertTrue(payload1_addr == payload2.data_ptr())

        payload3 = memory_cache.pop_or_allocate(self.compute_device, 10, torch.float)
        self.assertTrue(payload1_addr != payload3.data_ptr())
        print("payload3 ", payload3.data_ptr())

        payload2_addr = payload2.data_ptr()
        memory_cache.push(payload2)
        memory_cache.push(payload3)

        payload4 = memory_cache.pop_or_allocate(
            self.compute_device,
            10,
            torch.float,
        )
        self.assertTrue(payload2_addr == payload4.data_ptr())


if __name__ == "__main__":
    unittest.main()
