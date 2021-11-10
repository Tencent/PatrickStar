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

from __future__ import annotations
from abc import ABC, abstractmethod
from queue import PriorityQueue
from patrickstar.manager.metronome import Metronome
from patrickstar.core.const import ChunkState


class ChunkEvictionPolicyFactory(ABC):
    def __init__(self, metronome: Metronome):
        self.chunk_access_dict = {}
        self.chunk_release_dict = {}
        self.metronome = metronome

    def trace_access(self, chunk_id, dev):
        """
        Trace access information of chunk_id.
        Only works for the warmup phase.
        args:
            chunk_id : the id of chunk
            dev : the device uses the chunk at the moment
        """
        if not self.metronome.is_warmup():
            return
        cur_mom = self.metronome.moment()
        if (chunk_id, dev) not in self.chunk_access_dict:
            self.chunk_access_dict[(chunk_id, dev)] = [cur_mom]
        else:
            self.chunk_access_dict[(chunk_id, dev)].append(cur_mom)
            self.chunk_access_dict[(chunk_id, dev)].sort()

    def trace_release(self, chunk_id, dev):
        """
        Trace release information of chunk_id.
        Only works for the warmup phase.
        args:
            chunk_id : the id of chunk
            dev : the device uses the chunk at the moment
        """
        if not self.metronome.is_warmup():
            return
        cur_mom = self.metronome.moment()
        if (chunk_id, dev) not in self.chunk_access_dict:
            self.chunk_release_dict[(chunk_id, dev)] = [cur_mom]
        else:
            self.chunk_release_dict[(chunk_id, dev)].append(cur_mom)
            self.chunk_release_dict[(chunk_id, dev)].sort()

    def _chunk_next_used_moment(self, chunk_id, dev):
        """
        The very next memonet chunk_id has to be placed on dev.
        """
        # warmup, every chunk has the same priority
        if self.metronome.is_warmup():
            return 0
        cur_mom = self.metronome.moment()
        total_mom = self.metronome._total_moment
        access_mom_list = self.chunk_access_dict[(chunk_id, dev)]
        for mom in access_mom_list:
            if mom > cur_mom:
                return mom
        return total_mom + access_mom_list[0]

    @abstractmethod
    def derive_eviction_list(self, id_to_chunk_map, required_room, target_device):
        NotImplemented


class LatestAccessChunkEvictionPolicy(ChunkEvictionPolicyFactory):
    def derive_eviction_list(self, id_to_chunk_map, need_bytes, target_device):
        """
        Evict the chunk latest to be accessed on the current device.
        """
        movable_chunk_info = []
        q = PriorityQueue()
        for chunk_id, chunk in id_to_chunk_map.items():
            if (
                chunk.get_device() is not None
                and chunk.get_device().type == target_device.type
                and chunk.get_state() != ChunkState.COMPUTE
                and not chunk.is_pin()
            ):
                # The next moment when this chunk was accessed.
                next_mom = self._chunk_next_used_moment(chunk_id, target_device)
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
            moved_bytes += id_to_chunk_map[chunk_id].get_payload_space()
            moved_list.append(chunk_id)
            if moved_bytes >= need_bytes:
                break

        # Raise error when failed to make enough room.
        if moved_bytes < need_bytes:
            raise RuntimeError(
                f"device {target_device} still needs {need_bytes / 1e6} MB, "
                f"but there is not enough space on it, only {moved_bytes / 1e6} MB available. "
                f"movable_chunk_info {movable_chunk_info}"
            )
        return moved_list


# TODO(jiaruifang) evict the chunk earliest to be used on the opposite dev.
# opposite dev = CPU if dev = GPU
# opposite dev = GPU if dev = CPU
