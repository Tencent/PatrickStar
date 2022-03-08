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
