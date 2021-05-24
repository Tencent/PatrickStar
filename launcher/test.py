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

init_distributed(dist_backend='gloo')
rank = torch.distributed.get_rank()
manager = HybridPSManager()
manager.init([10, 10], [10, 10])

# 每个进程都分配一个chunk
chunk = Chunk(2, torch.half, 0)
chunk.allocate_payload(torch.device('cpu:0'))
chunk.payload.random_()

# 不同进程chunk执行allgather，每个进程获得一个allgather的chunk
print(f'before allgather, rank {rank}', chunk.payload)

chunk.allgather()

print(f'after allgather, rank {rank}', chunk.payload)

chunk.reduce_scatter()

print(f'after reduce_scatter, rank {rank}', chunk.payload)
