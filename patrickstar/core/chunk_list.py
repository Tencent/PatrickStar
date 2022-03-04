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

import logging

import torch

from patrickstar.core.chunk import Chunk
from patrickstar.core.const import ChunkState
from patrickstar.core.eviction_policy import ChunkEvictionPolicyBase
from patrickstar.core.memtracer import RuntimeMemTracer
from patrickstar.utils import logger, log_dist, global_timer


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
        self.chunk_size = chunk_size

        self._time_profile = True
        self.moments_cnt_of_iteration = None
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.chunk_eviction_policy = chunk_eviction_policy
        self.memory_tracer = memory_tracer

    def __getitem__(self, chunk_id: int):
        r"""Search a chunk by id."""
        return self.chunks[chunk_id]

    def __len__(self) -> int:
        return len(self.chunks)

    def get_chunk_memory_used(self, device):
        r"""The total memory of payload of all chunks on `device`.

        Args:
            device: :class:`torch.device`.
        Returns:
            float.
        """
        mem_used = 0
        for chunk in self.chunks:
            if (
                chunk.get_device() is not None
                and chunk.get_device().type == device.type
            ):
                mem_used += chunk.get_payload_space()
        return mem_used

    def try_allocate_payload(self, chunk: Chunk, compute_device):
        """
        Try our best to allocate payload for chunk.
        First free up chunk size space on the target device.
        If it dose not work, we second free up all chunks not in used on the target device.
        """
        self.prepare_device(compute_device, chunk.get_chunk_space())
        chunk.allocate_payload(compute_device)

    def access_chunk(self, chunk, compute_device):
        r"""Prepare the memory of chunk to `compute_device` with `chunk_id`.

        Args:
            chunk_id: int.
            compute_device: :class:`torch.device`.
        """
        if chunk.get_state() == ChunkState.RELEASED:
            self.try_allocate_payload(chunk, compute_device)
        elif chunk.get_device().type != compute_device.type:
            self.prepare_device(compute_device, chunk.get_chunk_space())
            chunk.move(compute_device)
        assert chunk.get_device().type == compute_device.type

    def prepare_device(self, target_device: torch.device, need_bytes: int):
        """
        Make `need_byes` room on `target_device`. If there are not enough empty
        space, we need to release or move away some chunks.

        Args:
            target_device: :class:`torch.device`.
            need_bytes: int.
        """
        if self._time_profile:
            global_timer.start_profile("CHUNK_LIST_prepare_device")

        ava_chunk_mem_size = self.memory_tracer.available_chunk_mem(target_device.type)
        remaining_chunk_mem_size = self.memory_tracer.remaining_chunk_mem(
            target_device.type
        )

        # TODO(jiaruifang) Situation where there is no space.
        # This condition is not good enough, we need to check if both CPU and GPU
        # don't have enough space.
        if ava_chunk_mem_size < need_bytes:
            log_dist(
                f"{target_device} has not enough space for {need_bytes} elements",
                level=logging.WARNING,
            )
            log_dist(
                f"{target_device} has not enough space for {need_bytes / 1e6} MB. "
                f"Device used Chunk Memory is {self.get_chunk_memory_used(target_device) / 1e6} MB. "
                f"Avaibale Chunk Memory is {ava_chunk_mem_size / 1e6} MB",
                level=logging.WARNING,
            )
            if self._time_profile:
                global_timer.finish_profile("CHUNK_LIST_prepare_device")
            return False

        need_bytes -= remaining_chunk_mem_size

        # No need for new allocation.
        if need_bytes <= 0:
            if self._time_profile:
                global_timer.finish_profile("CHUNK_LIST_prepare_device")
            return

        # Make some room on `target_device`.
        moved_list = self.chunk_eviction_policy.derive_eviction_list(
            self.chunks, need_bytes, target_device
        )

        # TODO(jiaruifang) Here we assume the new device has enough room and force the chunk
        # to new device. However, the size of the chunk may be smaller than the ava_chunk_mem
        # of the new device and trigger bugs.
        new_device = (
            torch.device("cpu") if target_device.type == "cuda" else self.device
        )

        # Move the chunk to new device. If there are not enough space on the new device, abort.
        for idx in moved_list:
            self.chunk_move(idx, new_device)

        if self._time_profile:
            global_timer.finish_profile("CHUNK_LIST_prepare_device")
        return True

    def chunk_move(self, chunk_id: int, device: torch.device):
        r"""Move chunk of id `chunk_id` to `device`.

        NOTE(): Please make sure `device` has enough remaining_chunk_mem before.

        Args:
            chunk_id: int.
            device: :class:`torch.device`.
        """
        if self._time_profile:
            global_timer.start_profile("CHUNK_LIST_chunk_move")

        chunk = self.chunks[chunk_id]

        remaining_chunk_mem_size = self.memory_tracer.remaining_chunk_mem(device.type)

        chunk_mem_size = chunk.get_payload_space()
        if remaining_chunk_mem_size < chunk_mem_size:
            raise RuntimeError(
                f"chunk move failed. {device} has not {chunk_mem_size / 1e6} MB memory space. "
                f"Free space is {remaining_chunk_mem_size / 1e6} MB. "
                f"The reason may be that the overall memory of CPU and GPU is not enough for the model."
            )
        if chunk.get_device() != device:
            logger.debug(f"move chunk {chunk_id} from {chunk.get_device()} to {device}")
            chunk.move(device)

        if self._time_profile:
            global_timer.finish_profile("CHUNK_LIST_chunk_move")

    def new_chunk(self, is_dummy: bool = False):
        r"""Create a chunk without initializing its memory.

        Args:
            chunk_id: int.
            chunk_size: int.
            is_dummy: bool.
        Returns:
            :class:`CommInfo`
        """
        chunk_id = len(self.chunks)
        chunk = Chunk(
            capacity=self.chunk_size,
            chunk_id=chunk_id,
            memory_tracer=self.memory_tracer,
            local_rank=self.local_rank,
            is_dummy=is_dummy,
        )
        self.chunks.append(chunk)
        return chunk
