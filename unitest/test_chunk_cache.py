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


class TestMemoryCache(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40

    def test_case1(self):
        self.compute_device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )

        memory_cache = MemoryCache(2)

        payload1 = memory_cache.allocate(self.compute_device, 10, torch.float, False)
        if payload1 is None:
            payload1 = torch.zeros(10, dtype=torch.float, device=self.compute_device)
        else:
            self.assertTrue(False)
        payload1_addr = payload1.data_ptr()
        memory_cache.recycle(payload1)
        payload1 = None

        payload2 = memory_cache.allocate(self.compute_device, 10, torch.float, False)
        print("payload2 ", payload2.data_ptr())
        self.assertTrue(payload1_addr == payload2.data_ptr())

        payload3 = memory_cache.allocate(self.compute_device, 10, torch.float, False)
        if payload3 is None:
            payload3 = torch.zeros(10, dtype=torch.float, device=self.compute_device)
        else:
            self.assertTrue(False)
        print("payload3 ", payload3.data_ptr())


if __name__ == "__main__":
    unittest.main()
