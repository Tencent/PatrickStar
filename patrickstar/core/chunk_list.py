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

from patrickstar.core.chunk import Chunk
from patrickstar.core.const import ChunkState
from patrickstar.core.eviction_policy import ChunkEvictionPolicyBase
from patrickstar.core.memtracer import RuntimeMemTracer


class ChunkList:
    r"""Manage the entities of all chunks.

    There are 4 kinds of chunk list:
        param fp16, param fp32, momentum, variance
    All of them are managed by one instance of this class.
    """

    def __init__(
        self,
        local_rank: int,
        memory_tracer: RuntimeMemTracer,
        chunk_eviction_policy: ChunkEvictionPolicyBase,
        chunk_size: int,
    ):
        self.chunks = []
        self.grad_chunks = []
        self.fp32_chunks = []
        self.chunk_size = chunk_size

        self.local_rank = local_rank
        self.chunk_eviction_policy = chunk_eviction_policy
        self.memory_tracer = memory_tracer

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, chunk_id):
        r"""Search a chunk by id."""
        return self.chunks[chunk_id]

    def new_chunk(self):
        r"""Create a chunk without initializing its memory."""
        chunk_id = len(self.chunks)
        chunk = Chunk(
            dtype=torch.half,
            capacity=self.chunk_size,
            chunk_id=chunk_id,
            memory_tracer=self.memory_tracer,
            local_rank=self.local_rank,
        )
        self.chunks.append(chunk)
        grad_chunk = Chunk(
            dtype=torch.half,
            capacity=self.chunk_size,
            chunk_id=chunk_id,
            memory_tracer=self.memory_tracer,
            local_rank=self.local_rank,
        )
        self.grad_chunks.append(grad_chunk)
        fp32_chunk = Chunk(
            dtype=torch.float,
            capacity=self.chunk_size,
            chunk_id=chunk_id,
            memory_tracer=self.memory_tracer,
            local_rank=self.local_rank,
        )
        self.fp32_chunks.append(fp32_chunk)
        return chunk

    def access_chunk(self, chunk, compute_device):
        r"""Prepare the memory of chunk to `compute_device` with `chunk_id`.

        Args:
            chunk_id: int.
            compute_device: :class:`torch.device`.
        """
        if chunk.get_state() == ChunkState.RELEASED:
            self.try_allocate_payload(chunk, compute_device)
        elif chunk.get_device().type != compute_device.type:
            self.chunk_eviction_policy.prepare_device(
                self, chunk.get_chunk_space(), compute_device
            )
            chunk.move(compute_device)
        assert chunk.get_device().type == compute_device.type

    def try_allocate_payload(self, chunk, compute_device):
        r"""
        Try our best to allocate payload for chunk.
        First free up chunk size space on the target device.
        If it dose not work, we second free up all chunks not in used on the target device.
        """
        self.chunk_eviction_policy.prepare_device(
            self, chunk.get_chunk_space(), compute_device
        )
        chunk.allocate_payload(compute_device)
