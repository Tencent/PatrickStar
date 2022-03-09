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


class ChunkList:
    def __init__(
        self,
        chunk_size,
        memtracer,
        chunk_eviction_policy,
    ):
        self.chunk_size = chunk_size
        # list of param.data chunks
        self.chunks = []
        # list of param.grad chunks, always on CPU
        self.grad_chunks = []
        # list of fp32 param.data backup, always on CPU
        self.fp32_chunks = []

        self.chunk_eviction_policy = chunk_eviction_policy
        self.memtracer = memtracer

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, chunk_id):
        return self.chunks[chunk_id]

    def new_chunk(self, is_dummy=False):
        r"""Create a param chunk as well as its grad chunk and fp32 chunk.

        Note that the payload of chunks are not allocated.
        """
        chunk_id = len(self.chunks)
        chunk = Chunk(
            dtype=torch.half,
            capacity=self.chunk_size,
            chunk_id=chunk_id,
            memtracer=self.memtracer,
            is_dummy=is_dummy,
        )
        self.chunks.append(chunk)
        grad_chunk = Chunk(
            dtype=torch.half,
            capacity=self.chunk_size,
            chunk_id=chunk_id,
            memtracer=self.memtracer,
            is_dummy=is_dummy,
        )
        self.grad_chunks.append(grad_chunk)
        fp32_chunk = Chunk(
            dtype=torch.float,
            capacity=self.chunk_size,
            chunk_id=chunk_id,
            memtracer=self.memtracer,
            is_dummy=is_dummy,
        )
        self.fp32_chunks.append(fp32_chunk)
        return chunk

    def access_chunk(self, chunk, compute_device):
        r"""Move or allocate `chunk` to `compute_device`."""
        if chunk.get_state() == ChunkState.RELEASED:
            self.try_allocate_payload(chunk, compute_device)
        elif chunk.get_device().type != compute_device.type:
            self.chunk_eviction_policy.prepare_device(
                self, chunk.get_chunk_space(), compute_device
            )
            chunk.move(compute_device)
        assert chunk.get_device().type == compute_device.type

    def try_allocate_payload(self, chunk, compute_device):
        r"""Allocate payload of `chunk`."""
        self.chunk_eviction_policy.prepare_device(
            self, chunk.get_chunk_space(), compute_device
        )
        chunk.allocate_payload(compute_device)
