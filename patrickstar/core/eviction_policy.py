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

from abc import ABC, abstractmethod
from queue import PriorityQueue

import torch

from patrickstar.core.const import ChunkState
from patrickstar.utils import logger


class ChunkEvictionPolicyBase(ABC):
    def __init__(self, local_rank, metronome, memory_tracer):
        self.local_rank = local_rank
        self.chunk_access_dict = {}
        self.metronome = metronome
        self.memory_tracer = memory_tracer

    def trace_access(self, chunk_id, device):
        """
        Trace access information of chunk_id.
        Only works for the warmup stage.
        args:
            chunk_id : the id of chunk
            device : the device uses the chunk at the moment
        """
        cur_mom = self.metronome.moment
        if (chunk_id, device) not in self.chunk_access_dict:
            self.chunk_access_dict[(chunk_id, device)] = []
        self.chunk_access_dict[(chunk_id, device)].append(cur_mom)

    def next_access_moment(self, chunk_id, device):
        """
        The very next memonet chunk_id has to be placed on device.
        """
        # warmup, every chunk has the same priority
        if self.metronome.is_warmup:
            return 0
        cur_mom = self.metronome.moment
        total_mom = self.metronome.total_moment

        if (chunk_id, device) not in self.chunk_access_dict:
            return 2 * total_mom
        access_mom_list = self.chunk_access_dict[(chunk_id, device)]
        for mom in access_mom_list:
            if mom > cur_mom:
                return mom
        return total_mom + access_mom_list[0]

    def prepare_device(self, chunks, required_room, target_device):
        remaining_chunk_mem_size = self.memory_tracer.remaining_chunk_mem(
            target_device.type
        )
        required_room -= remaining_chunk_mem_size

        if required_room <= 0:
            return

        moved_list = self.derive_eviction_list(chunks, required_room, target_device)

        if target_device.type == "cuda":
            new_device = torch.device("cpu:0")
        else:
            new_device = torch.device(f"cuda:{self.local_rank}")

        for chunk_id in moved_list:
            chunk = chunks[chunk_id]
            assert chunk.get_device() != new_device
            chunk.move(new_device)

    @abstractmethod
    def derive_eviction_list(self, chunks, required_room, target_device):
        raise NotImplementedError("derive_eviction_list is not Implemented")


class LRUEvictionPolicy(ChunkEvictionPolicyBase):
    def derive_eviction_list(self, chunks, need_bytes, target_device):
        """
        Evict the chunk latest to be accessed on the current device.
        """
        movable_chunk_info = []
        q = PriorityQueue()
        for chunk_id, chunk in enumerate(chunks):
            if (
                chunk.get_device() is not None
                and chunk.get_device().type == target_device.type
                and chunk.get_state() != ChunkState.COMPUTE
                and chunk.get_state() != ChunkState.RELEASED
                and not chunk.is_pin()
            ):
                # The next moment when this chunk was accessed.
                next_mom = self.next_access_moment(chunk_id, target_device)
                # Order by `next_mom`s, from large to small
                # and by chunk_ids if `next_mom` are the same (only happens during warmup).
                q.put((-next_mom, chunk_id))
                movable_chunk_info.append(f"{next_mom}_{chunk_id}")
            # TODO(jiaruifang) Do not release `FREE` chunks immediately for reuse.
            # assert chunk.get_state() != ChunkState.FREE
        moved_list = []
        moved_bytes = 0
        while not q.empty():
            next_mom, chunk_id = q.get()
            moved_bytes += chunks[chunk_id].get_payload_space()
            moved_list.append(chunk_id)
            if moved_bytes >= need_bytes:
                break

        # Raise error when failed to make enough room.
        if moved_bytes < need_bytes:
            logger.warning(
                f"device {target_device} still needs {need_bytes / 1e6} MB, "
                f"but there is not enough space on it, only {moved_bytes / 1e6} MB available. "
                f"movable_chunk_info {movable_chunk_info}",
            )
        return moved_list
