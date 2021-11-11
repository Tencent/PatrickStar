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
from patrickstar.utils.memory_monitor import get_sys_memory_used
from patrickstar.core.memtracer.memtracer import AsyncMemoryMonitor


class TestAsynMemoryMonitor(unittest.TestCase):
    def setUp(self):
        pass

    def helper_func(self):
        dev = torch.device("cuda:0")
        m = 400
        n = 500
        k = 600
        a = torch.randn(m, k, device=torch.device("cuda:0"))
        b = torch.randn(k, n, device=torch.device("cuda:0"))
        c = torch.randn(m, n, device=torch.device("cuda:0"))
        print(f"mem usage before matmul: {get_sys_memory_used(dev)}")
        start_mem = get_sys_memory_used(dev)
        for i in range(10):
            c += torch.matmul(a, b)
        print(f"mem usage after matmul: {get_sys_memory_used(dev)}")
        finish_mem = get_sys_memory_used(dev)
        return max(start_mem, finish_mem)

    def test_async_mem_monitor(self):
        mem_monitor = AsyncMemoryMonitor()
        mem_monitor.start()
        max_mem_coarse = self.helper_func()
        max_mem_fine = mem_monitor.finish()
        self.assertTrue(max_mem_fine >= max_mem_coarse)
        # max_mem fine 3760640, corse 2960384
        # indicates the operator will generate singnificant temp buff.
        print(f"max_mem fine {max_mem_fine}, corse {max_mem_coarse}")


if __name__ == "__main__":
    unittest.main()
