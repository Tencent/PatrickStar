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

import torch
import torch.distributed as dist
from common import distributed_test
import time
import logging

from manager import HybridPSManager
from client import HybridPSClient, AccessType
from utils import see_memory_usage

manager = HybridPSManager()


def test_monitor():
    # gpu 4GB, CPU 4 GB
    manager.init(gpu_info=[1024 * 1024 * 1024 * 4],
                 cpu_info=[4 * 1024 * 1024 * 1024])
    print("is init manager", HybridPSManager().is_init())

    # nothing in memory
    # MA 0.0 KB         Max_MA 0.0 KB         CA 0.0 KB         Max_CA 0 KB
    # CPU Virtual Memory:  used = 1.64 GB, percent = 10.5%
    see_memory_usage('just init manager', force=True)

    check_pytorch_runtime = True
    if check_pytorch_runtime:
        t1 = torch.randn(1, device=torch.device('cuda'))

        # 只要用一下gpu tensor，那么CPU立刻分配近4GB的内存出来。
        # MA 0.5 KB         Max_MA 0.5 KB         CA 2048.0 KB         Max_CA 2048 KB
        # CPU Virtual Memory:  used = 3.94 GB, percent = 25.2%
        see_memory_usage(
            'after use PyTorch runtime to allocate a very small piece of GPU memory',
            force=True)

    param1 = torch.nn.Parameter(torch.randn(
        512,
        device=torch.cuda.current_device()
        if torch.cuda.is_available() else torch.device('cpu')),
                                requires_grad=False)

    see_memory_usage('after use PyTorch runtime allocate param1', force=True)
    param2 = torch.nn.Parameter(torch.randn(
        512,
        device=torch.cuda.current_device()
        if torch.cuda.is_available() else torch.device('cpu')),
                                requires_grad=False)

    # now 4KB GPU memory
    # MA 4.0 KB         Max_MA 4.0 KB         CA 2048.0 KB         Max_CA 2048 KB
    # CPU Virtual Memory:  used = 3.92 GB, percent = 25.1%
    # NOTE(jiaruifang) Why CPU memory usage increase? I guess the PyTorch runtime occupy some memory.
    see_memory_usage('after init param1 and parm2', force=True)

    client = HybridPSClient(rank=0, default_chunk_size=1024)

    client.register_param(param1)

    # 应该也是 4KB
    see_memory_usage('after register param1', force=True)
    client.register_param(param2)

    # now 4MB CPU memory, 0KB GPU memory
    # MA 0.0 KB         Max_MA 4.0 KB         CA 2048.0 KB         Max_CA 2048 KB
    # CPU Virtual Memory:  used = 3.92 GB, percent = 25.1%
    see_memory_usage('after register param1 and parm2', force=True)

    assert param1.device == torch.device('cpu')

    # 测试chunk内存复用

    # 测试chunk的删除
    client.register_param(param1, )


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)

    test_monitor()
