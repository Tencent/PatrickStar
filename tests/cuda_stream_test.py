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
import os
from simple_net import SimpleModel

import time

cpu_device = torch.device('cpu')
device = torch.device('cuda:0')


def blocking_helper(size):
    torch.manual_seed(0)
    inputs = []
    for i in range(4):
        inputs.append(torch.randn(size))

    start_time = time.time()
    for i in range(4):
        inputs[i].to(device)
        inputs[i] = torch.sum(inputs[i])
    print(f"blocking_helper {size/1e3}K elapse ", time.time() - start_time)

    for i in range(4):
        print(i, torch.sum(inputs[i]))


def non_blocking_helper(size):
    torch.manual_seed(0)
    inputs = []
    for i in range(4):
        inputs.append(torch.randn(size).pin_memory())

    start_time = time.time()
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(copy_stream):
        inputs[0].to(device, non_blocking=True)

    torch.cuda.synchronize()
    with torch.cuda.stream(copy_stream):
        inputs[1].to(device, non_blocking=True)

    with torch.cuda.stream(compute_stream):
        inputs[0] = torch.sum(inputs[0])
    torch.cuda.synchronize()
    with torch.cuda.stream(copy_stream):
        inputs[2].to(device, non_blocking=True)
    with torch.cuda.stream(compute_stream):
        inputs[1] = torch.sum(inputs[1])

    torch.cuda.synchronize()
    with torch.cuda.stream(copy_stream):
        inputs[3].to(device, non_blocking=True)
    with torch.cuda.stream(compute_stream):
        inputs[2] = torch.sum(inputs[2])

    # torch.cuda.synchronize()
    with torch.cuda.stream(compute_stream):
        inputs[3] = torch.sum(inputs[3])

    print(f"non blocking_helper {size/1e3}K  elapse ",
          time.time() - start_time)
    for i in range(4):
        print(i, torch.sum(inputs[i]))


def non_blocking_helper_v1(size):
    torch.manual_seed(0)
    inputs = []
    for i in range(4):
        inputs.append(torch.randn(size).pin_memory())

    start_time = time.time()
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    with torch.cuda.stream(copy_stream):
        inputs[0].to(device, non_blocking=True)

    torch.cuda.synchronize()
    with torch.cuda.stream(copy_stream):
        inputs[1].to(device, non_blocking=True)

    inputs[0] = torch.sum(inputs[0])
    torch.cuda.synchronize()
    with torch.cuda.stream(copy_stream):
        inputs[2].to(device, non_blocking=True)
    inputs[1] = torch.sum(inputs[1])

    torch.cuda.synchronize()
    with torch.cuda.stream(copy_stream):
        inputs[3].to(device, non_blocking=True)
    inputs[2] = torch.sum(inputs[2])

    # torch.cuda.synchronize()
    inputs[3] = torch.sum(inputs[3])

    print(f"non blocking_helper v1 {size/1e3}K  elapse ",
          time.time() - start_time)
    for i in range(4):
        print(i, torch.sum(inputs[i]))


def non_blocking_helper_v2(size):
    torch.manual_seed(0)
    inputs = []
    for i in range(4):
        inputs.append(torch.randn(size))
    copy_stream = torch.cuda.Stream()

    start_time = time.time()
    for i in range(4):
        with torch.cuda.stream(copy_stream):
            inputs[i].to(device, non_blocking=True)

    torch.cuda.synchronize()
    for i in range(4):
        inputs[i] = torch.sum(inputs[i])
    print(f"blocking_helper {size/1e3}K elapse ", time.time() - start_time)

    for i in range(4):
        print(i, torch.sum(inputs[i]))


if __name__ == "__main__":

    size = 1000000

    for size in [1000, 100000, 1000000, 10000000]:
        print("========")
        blocking_helper(size)
        blocking_helper(size)

        non_blocking_helper(size)
        non_blocking_helper(size)

        non_blocking_helper_v1(size)
        non_blocking_helper_v1(size)

        non_blocking_helper_v2(size)
        non_blocking_helper_v2(size)
