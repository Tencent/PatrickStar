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

from client import ChunkList, PSTensorStatus
import logging
import torch
from manager import HybridPSManager


def test_basic():
    manager = HybridPSManager()
    manager.reset([32, 32], [1024])
    data_type = torch.float
    chunk_list = ChunkList(default_chunk_size=128)
    # 之前分配的chunk中尝试分配10空间
    chunk_id = chunk_list.least_used_chunk()
    assert chunk_id == 0

    chunk_id, tensor = chunk_list.allocate(10, data_type, 0)
    assert chunk_id == 0

    chunk_id, tensor = chunk_list.allocate(100, data_type, 1)
    assert chunk_id == 0
    chunk_id = chunk_list.least_used_chunk()
    chunk_list[chunk_id].visit()

    chunk_id, tensor = chunk_list.allocate(100, data_type, 2)
    assert (chunk_id == 1)

    chunk_id = chunk_list.least_used_chunk()
    assert chunk_id == 1


def test_reuse():
    manager = HybridPSManager()
    manager.reset([32, 32], [1024])
    data_type = torch.float
    chunk_list = ChunkList(default_chunk_size=20)
    chunk_id, tensor = chunk_list.allocate(16, data_type, 0)
    assert chunk_id == 0

    chunk_id, tensor = chunk_list.allocate(4, data_type, 1)
    assert chunk_id == 0

    chunk_list[0].tensor_info_list.set_status(0, PSTensorStatus.FREE)

    chunk_id, tensor = chunk_list.allocate(16, data_type, 2)
    assert chunk_id == 0


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    # test_basic()
    test_reuse()

    # 再分配一个chunk
    # chunk_list.new_chunk(128)
