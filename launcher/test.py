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
"""
run this script with
python runner.py --num_nodes 1 --num_gpus 2 test.py
"""

from client import Chunk
from manager import HybridPSManager
from utils import init_distributed
import torch
import torch.distributed as dist


def test_collective_comm():
    rank = torch.distributed.get_rank()
    manager = HybridPSManager()
    manager.init([10, 10], [10, 10])

    # 每个进程都分配一个chunk
    chunk = Chunk(4, torch.half, 0)
    chunk.allocate_payload(torch.device(f'cuda:{rank}'))
    chunk.payload.random_()

    # 不同进程chunk执行allgather，每个进程获得一个allgather的chunk
    print(f'before allgather, rank {rank}', chunk.payload)

    chunk.allgather()

    print(f'after allgather, rank {rank}', chunk.payload)

    chunk.reduce_scatter()

    print(f'after reduce_scatter, rank {rank}', chunk.payload)


def test_p2p():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    tensor = torch.ones(10)
    if rank == 0:
        tensor = torch.ones(10) + 123
        for i in range(1, world_size):
            dist.send(tensor=tensor, dst=i)
    else:
        dist.recv(tensor=tensor, src=0)

    print(tensor)


if __name__ == "__main__":
    init_distributed(dist_backend='gloo')
    test_p2p()
