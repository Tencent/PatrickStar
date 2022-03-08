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

    def prepare_device(self, chunk_list, required_room, target_device):
        remaining_chunk_mem_size = self.memory_tracer.remaining_chunk_mem(
            target_device.type
        )
        required_room -= remaining_chunk_mem_size

        if required_room <= 0:
            return

        chunks_to_move = self.derive_eviction_list(
            chunk_list, required_room, target_device
        )

        if target_device.type == "cuda":
            new_device = torch.device("cpu:0")
        else:
            new_device = torch.device(f"cuda:{self.local_rank}")

        for chunk in chunks_to_move:
            assert chunk.get_device() != new_device
            chunk.move(new_device)

    @abstractmethod
    def derive_eviction_list(self, chunks, required_room, target_device):
        raise NotImplementedError("derive_eviction_list is not Implemented")


class LRUEvictionPolicy(ChunkEvictionPolicyBase):
    def derive_eviction_list(self, chunk_list, need_bytes, target_device):
        """
        Evict the chunk latest to be accessed on the current device.
        """
        chunks_to_move = []
        moved_bytes = 0
        q = PriorityQueue()
        for chunk_id, chunk in enumerate(chunk_list.chunks):
            if (
                chunk.get_state() == ChunkState.HOLD
                and chunk.get_device().type == target_device.type
                and not chunk.is_pin()
            ):
                # The next moment when this chunk was accessed.
                next_mom = self.next_access_moment(chunk_id, target_device)
                # Order by `next_mom`s, from large to small
                # and by chunk_ids if `next_mom` are the same (only happens during warmup).
                q.put((-next_mom, chunk_id))
            # TODO(jiaruifang) Do not release `FREE` chunks immediately for reuse.
            # assert chunk.get_state() != ChunkState.FREE
        while not q.empty():
            next_mom, chunk_id = q.get()
            chunk = chunk_list.chunks[chunk_id]
            moved_bytes += chunk.get_payload_space()
            chunks_to_move.append(chunk)
            # move grad chunk together with data chunk
            grad_chunk = chunk_list.grad_chunks[chunk_id]
            if (
                grad_chunk.get_state() == ChunkState.HOLD
                and grad_chunk.get_device().type == target_device.type
            ):
                moved_bytes += grad_chunk.get_payload_space()
                chunks_to_move.append(grad_chunk)

            if moved_bytes >= need_bytes:
                break

        # Raise error when failed to make enough room.
        if moved_bytes < need_bytes:
            logger.warning(
                f"device {target_device} still needs {need_bytes / 1e6} MB, "
                f"but there is not enough space on it, only {moved_bytes / 1e6} MB available."
            )
        return chunks_to_move
