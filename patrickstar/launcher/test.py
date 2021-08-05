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

from patrickstar.utils import init_distributed
import torch
import torch.distributed as dist
import time
from patrickstar.utils import see_memory_usage, get_sys_memory_used


def test_reduce_scatter(numel, repeat=10):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f'cuda:{rank}')
    payload = torch.randn(numel, dtype=torch.half, device=device)
    input_list = []
    group_list = []
    for i in range(world_size):
        if i == rank:
            input_list.append(payload)
        else:
            input_list.append(
                torch.randn(numel, dtype=torch.half, device=device))
        group_list.append(i)
    # group = torch.distributed.new_group(group_list)

    if rank == 0:
        print(
            f'before reduce_scatter gpu mem {get_sys_memory_used(device)/1e6} MN'
        )
    start_time = time.time()
    for i in range(repeat):
        torch.distributed.reduce_scatter(
            payload,
            input_list,
            op=torch.distributed.ReduceOp.SUM,
            #group=group,
            async_op=False)

    elapse = time.time() - start_time
    input_list = []
    if rank == 0:
        print(
            f'after reduce_scatter gpu mem {get_sys_memory_used(device)/1e6} MB'
        )
        print(
            f"rank {rank} test reduce_scatter finished {numel/1024/1024} MB {world_size* numel*2*repeat/1e6/elapse} MB/s"
        )


def test_allgather(numel, repeat=10, is_async=False):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    device = torch.device(f'cuda:{rank}')
    payload = torch.randn(numel, dtype=torch.half, device=device)
    input_list = []
    group_list = []
    for i in range(world_size):
        if i == rank:
            input_list.append(payload)
        else:
            input_list.append(
                torch.zeros(numel, dtype=torch.half, device=device))
        group_list.append(i)
    group = torch.distributed.new_group(group_list)
    if rank == 0:
        print(f'before allgather gpu mem {get_sys_memory_used(device)/1e6} MB')
    start_time = time.time()
    for i in range(repeat):
        handle = torch.distributed.all_gather(
            input_list,
            payload,
            #group=group,
            async_op=is_async)

    elapse = time.time() - start_time
    if rank == 0:
        print(f'after allgather gpu mem {get_sys_memory_used(device)/1e6} MB')
        print(
            f"rank {rank} test allgather finished {numel/1024/1024} MB {world_size* numel*2*repeat/1e6/elapse} MB/s"
        )

    payload_new = torch.randn(numel, dtype=torch.half, device=device)
    input_list_new = []
    for i in range(world_size):
        if i == rank:
            input_list_new.append(payload_new)
        else:
            input_list_new.append(
                torch.zeros(numel, dtype=torch.half, device=device))
    torch.cuda.empty_cache()
    start_time = time.time()
    for i in range(repeat):
        handle = torch.distributed.all_gather(
            input_list_new,
            payload_new,
            #group=group,
            async_op=is_async)

    elapse = time.time() - start_time
    input_list = []
    if rank == 0:
        print(
            f'[repeat] after allgather gpu mem {get_sys_memory_used(device)/1e6} MB'
        )
        print(
            f"[repeat] rank {rank} test allgather finished {numel/1024/1024} MB {world_size* numel*2*repeat/1e6/elapse} MB/s"
        )

    return handle


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
    init_distributed(dist_backend='nccl')
    # cpu_comm_group = torch.distributed.new_group(backend='gloo')
    # test_p2p()
    handle_list = []
    handle = test_allgather(128 * 1024 * 1024, repeat=1, is_async=False)
    handle_list.append(handle)
    handle = test_allgather(128 * 1024 * 1024, repeat=1, is_async=False)
    handle_list.append(handle)
    handle = test_allgather(128 * 1024 * 1024, repeat=3, is_async=False)
    handle_list.append(handle)
    for handle in handle_list:
        if handle is not None:
            handle.wait()
    test_allgather(128 * 1024 * 1024, repeat=4)
    # test_reduce_scatter(512*1024*1024)
    # for numel in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     test_reduce_scatter(numel*1024*1024)
    # for numel in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
    #     test_allgather(numel*1024*1024)
