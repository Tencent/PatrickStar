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

from typing import List

import torch

from patrickstar.core.const import ChunkType
from patrickstar.core.memtracer import RuntimeMemTracer
from patrickstar.profiler import profiler
from patrickstar.utils import logger, get_rank, get_world_size
import patrickstar.utils.global_timer as global_timer
from .chunk_data import Chunk
from .comm import CommInfo
from .const import ChunkState
from patrickstar.core.eviction_policy import ChunkEvictionPolicyBase


class ChunkList(object):
    r"""Manage the entities of all chunks.

    There are 4 kinds of chunk list:
        param fp16, param fp32, momentum, variance
    All of them are managed by one instance of this class.
    """

    generated_chunk_id = -1

    def __init__(
        self,
        local_rank: int,
        memory_tracer: RuntimeMemTracer,
        chunk_eviction_policy: ChunkEvictionPolicyBase,
    ):
        """
        Args:
            local_rank: int.
        """
        self.id_to_chunk_map: dict[int, Chunk] = {}
        self.chunk_type_to_id_list_map: dict[ChunkType, int] = {}
        for chunk_type in ChunkType:
            self.chunk_type_to_id_list_map[chunk_type] = []

        self._time_profile = True
        self.moments_cnt_of_iteration = None
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.chunk_eviction_policy = chunk_eviction_policy
        self.memory_tracer = memory_tracer

    def chunk_ids_generator(self, chunk_type: ChunkType):
        r"""Return the chunk_id of all chunks with type `chunk_type`

        Args:
            chunk_type: :class:`ChunkType`.
        """
        for chunk_id in self.chunk_type_to_id_list_map[chunk_type]:
            yield chunk_id

    def generate_chunk_id(self) -> int:
        r"""Get the chunk id of next chunk."""
        ChunkList.generated_chunk_id += 1
        return ChunkList.generated_chunk_id

    def __getitem__(self, chunk_id: int):
        r"""Search a chunk by id."""
        return self.id_to_chunk_map.get(chunk_id)

    def size(self) -> int:
        r"""Total number of chunks."""
        return len(self.id_to_chunk_map)

    def __len__(self) -> int:
        return self.size()

    def get_chunk_memory_used(self, device):
        r"""The total memory of payload of all chunks on `device`.

        Args:
            device: :class:`torch.device`.
        Returns:
            float.
        """
        mem_used = 0
        for _, chunk in self.id_to_chunk_map.items():
            if (
                chunk.get_device() is not None
                and chunk.get_device().type == device.type
            ):
                mem_used += chunk.get_payload_space()
        return mem_used

    def max_chunk_size(self):
        max_size = 0
        for _, chunk in self.id_to_chunk_map.items():
            max_size = max(chunk.capacity, max_size)
        return max_size

    def access_chunk(self, chunk_id: int, compute_device: torch.device):
        r"""Prepare the memory of chunk to `compute_device` with `chunk_id`.

        1. standalone mode
        In standalone mode, we need to move the chunk when it is on other devices.
            TODO(jiaruifang) Add async copy and record the lifecycle of chunks during
            the first iteration, so that we can prefetch the next chunk after sync
            the memcopy of the first chunk.
        2. distributed mode
        Use allgather to fetch chunks from other processes.

        Args:
            chunk_id: int.
            compute_device: :class:`torch.device`.
        """

        chunk = self.id_to_chunk_map[chunk_id]

        chunk_state = chunk.get_state()

        payload_space = chunk.get_chunk_space()

        # If chunk was released, we need to reallocate it.
        # In distributed mode, we need a global payload.
        if chunk_state == ChunkState.RELEASED:
            logger.debug(
                f"rank {get_rank()} access_chunk chunk {chunk_id}, "
                f"need to allocate {payload_space} B memory on {compute_device}"
            )

            # TODO(jiaruifang) We need to prepare for the distributed environment.
            self.prepare_device(compute_device, payload_space)
            if chunk.allocate_payload(compute_device):
                return
            else:
                self.prepare_device(compute_device, 2 * payload_space)
                if chunk.allocate_payload(compute_device) is False:
                    raise RuntimeError(
                        f"Allocate Payload Failed even if we have moved out more memory from {compute_device}"
                    )
            return
        elif chunk.get_device().type != compute_device.type:
            self.prepare_device(compute_device, payload_space)
            chunk.move(compute_device)
            assert (
                chunk.get_device().type == compute_device.type
            ), f"chunk device {chunk.get_device()} compute device {compute_device}"
            return
        else:
            logger.debug(f"access_chunk chunk {chunk_id} already on {compute_device}")

    def prepare_device(self, target_device: torch.device, need_bytes: int):
        """
        Make `need_byes` room on `target_device`. If there are not enough empty
        space, we need to release or move away some chunks.

        Args:
            target_device: :class:`torch.device`.
            need_bytes: int.
        """
        if self._time_profile:
            global_timer.my_timer.start_profile("CHUNK_LIST_prepare_device")

        ava_chunk_mem_size = self.memory_tracer.available_chunk_mem(target_device.type)
        remaining_chunk_mem_size = self.memory_tracer.remaining_chunk_mem(
            target_device.type
        )

        logger.debug(
            f"prepare_target: device {target_device} need_bytes {need_bytes / 1e6} MB, "
            f"ava_chunk_mem_size {ava_chunk_mem_size / 1e6} MB, "
            f"remaining_chunk_mem_size {remaining_chunk_mem_size / 1e6} MB."
        )

        # TODO(jiaruifang) Situation where there is no space.
        # This condition is not good enough, we need to check if botn CPU and GPU
        # don't have enough space.
        if ava_chunk_mem_size < need_bytes:
            logger.error(
                f"{target_device} has not enough space for {need_bytes} elements"
            )
            # TODO(jiaruifang) We can catch the error and the release or move the chunks here.
            raise RuntimeError(
                f"{target_device} has not enough space for {need_bytes / 1e6} MB. "
                f"Device used Chunk Memory is {self.get_chunk_memory_used(target_device) / 1e6} MB. "
                f"Avaibale Chunk Memory is {ava_chunk_mem_size / 1e6} MB"
            )

        extra_need_bytes = need_bytes - remaining_chunk_mem_size

        logger.debug(
            f"{target_device} (ava_chunk_mem_size {ava_chunk_mem_size / 1e6} MB) "
            f"now remaining_chunk_mem_size size {remaining_chunk_mem_size / 1e6} MB, "
            f"needs {need_bytes / 1e6} MB"
        )
        # No need for new allocation.
        if extra_need_bytes <= 0:
            if self._time_profile:
                global_timer.my_timer.finish_profile("CHUNK_LIST_prepare_device")
            return

        logger.debug(
            f"the device {target_device} has no enough free chunk memory, "
            f"required size is {extra_need_bytes} bytes"
        )
        # Make some room on `target_device`.
        moved_list = self._chunk_to_move_out_for_room_making(
            extra_need_bytes, target_device
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
            global_timer.my_timer.finish_profile("CHUNK_LIST_prepare_device")

    def make_room(self, offload_size_in_bytes, target_device):
        r"""Move `offload_size_in_bytes` size of chunks away from `target_device`.

        Can not move chunk of state `COMPUTE`.

        Args:
            offload_size_in_bytes: int.
            target_device: :class:`torch.device`.
        """
        if self._time_profile:
            global_timer.my_timer.start_profile("CHUNK_LIST_make_room")

        moved_list = self._chunk_to_move_out_for_room_making(
            offload_size_in_bytes, target_device
        )

        new_device = (
            torch.device("cpu") if target_device.type == "cuda" else self.device
        )

        for idx in moved_list:
            self.chunk_move(idx, new_device)

        if self._time_profile:
            global_timer.my_timer.finish_profile("CHUNK_LIST_make_room")

    def chunk_move(self, chunk_id: int, device: torch.device):
        r"""Move chunk of id `chunk_id` to `device`.

        NOTE(): Please make sure `device` has enough remaining_chunk_mem before.

        Args:
            chunk_id: int.
            device: :class:`torch.device`.
        """
        if self._time_profile:
            global_timer.my_timer.start_profile("CHUNK_LIST_chunk_move")

        chunk = self.id_to_chunk_map[chunk_id]

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
            global_timer.my_timer.finish_profile("CHUNK_LIST_chunk_move")

    def new_chunk(
        self,
        chunk_id: int,
        chunk_size: int,
        data_type: torch.dtype,
        is_dummy: bool = False,
        chunk_type: ChunkType = ChunkType.UNDEF,
    ):
        r"""Create a chunk without initializing its memory.

        Args:
            chunk_id: int.
            chunk_size: int.
            data_type: :class:`torch.dtype`.
            is_dummy: bool.
            chunk_type: :class:ChunkType.
        Returns:
            :class:`CommInfo`
        """
        if chunk_id in self.id_to_chunk_map:
            raise RuntimeError(
                f"chunk list new chunk with chunk_id {chunk_id} already existed"
            )
        self.id_to_chunk_map[chunk_id] = Chunk(
            capacity=chunk_size,
            data_type=data_type,
            chunk_id=chunk_id,
            memory_tracer=self.memory_tracer,
            local_rank=self.local_rank,
            is_dummy=is_dummy,
        )
        world_size = get_world_size()
        global_rank = get_rank()
        self.chunk_type_to_id_list_map[chunk_type].append(chunk_id)
        if profiler.started():
            profiler.chunk_life_cycle[chunk_id] = {"type": chunk_type, "life_cycle": []}
        num_type_chunk = len(self.chunk_type_to_id_list_map[chunk_type])
        comm_info = CommInfo(
            chunk_type=chunk_type,
            group_id=(num_type_chunk - 1) // world_size,
            offset=(num_type_chunk - 1) % world_size,
        )
        logger.debug(
            f"global_rank {global_rank}, allocate with new chunk chunk_id {chunk_id} size {chunk_size} "
            f"data_type {data_type} comm group {comm_info}"
        )
        return comm_info

    def is_empty(self, chunk_type: ChunkType):
        r"""Whether chunk list of type `chunk_type` is empty."""
        return len(self.chunk_type_to_id_list_map[chunk_type]) == 0

    def last_chunk_id(self, chunk_type: ChunkType):
        r"""Get the last id of type `chunk_type`."""
        if self.is_empty(chunk_type):
            raise RuntimeError(
                f"Call last_chunk_id on an empty {chunk_type} chunk list"
            )
        return self.chunk_type_to_id_list_map[chunk_type][-1]

    def generate_chunk(self):
        r"""Return all the chunks along with its id."""
        for chunk_id, chunk in self.id_to_chunk_map.items():
            yield chunk_id, chunk

    def _chunk_to_move_out_for_room_making(
        self, size_in_bytes: int, target_device: torch.device
    ) -> List:
        r"""Find the chunks to move for making `size_in_bytes` of room on `target_device`.

        Args:
            size_in_bytes: int.
            target_device: :class:`torch.device`.
        Returns:
            A list of chunk_ids.
        """
        moved_list = self.chunk_eviction_policy.derive_eviction_list(
            self.id_to_chunk_map, size_in_bytes, target_device
        )
        return moved_list

    def update_state(self, chunk_id, old_state, new_state):
        r"""Update the state of chunk of id `chunk_id`."""
        self.id_to_chunk_map[chunk_id].update_state(old_state, new_state)
