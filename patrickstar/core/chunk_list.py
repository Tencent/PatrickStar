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
            capacity=self.chunk_size,
            chunk_id=chunk_id,
            memory_tracer=self.memory_tracer,
            local_rank=self.local_rank,
        )
        self.chunks.append(chunk)
        grad_chunk = Chunk(
            capacity=self.chunk_size,
            chunk_id=chunk_id + 1000,
            memory_tracer=self.memory_tracer,
            local_rank=self.local_rank,
        )
        self.grad_chunks.append(grad_chunk)
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
                self.chunks, chunk.get_chunk_space(), compute_device
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
            self.chunks, chunk.get_chunk_space(), compute_device
        )
        chunk.allocate_payload(compute_device)
