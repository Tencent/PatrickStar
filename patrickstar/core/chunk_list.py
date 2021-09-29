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

from queue import PriorityQueue
from typing import List

import torch

from patrickstar.core.const import ChunkListType
from patrickstar.manager import PatrickStarManager
from patrickstar.profiler import profiler
from patrickstar.utils import logger, get_rank, get_world_size
import patrickstar.utils.global_timer as global_timer
from .chunk_data import Chunk
from .const import PSChunkStatus


class ChunkList(object):
    """
    There are 4 kinds of chunk list:
        param fp16, param fp32, momentum, variance
    All of them are managed by one instance of this class.
    """

    generated_chunk_id = -1

    def __init__(self, local_rank: int):
        self.chunk_id_to_chunk_dict_map: dict[int, Chunk] = {}
        self.chunk_type_to_id_list_map: dict[ChunkListType, int] = {}
        for chunk_type in ChunkListType:
            self.chunk_type_to_id_list_map[chunk_type] = []

        self._time_profile = True
        self.moments_cnt_of_iteration = None
        self.local_rank = local_rank

    def chunk_ids_generator(self, chunk_list_type: ChunkListType):
        """
        Return the chunk_id of all chunks with type `chunk_list_type`
        """
        for chunk_id in self.chunk_type_to_id_list_map[chunk_list_type]:
            yield chunk_id

    def generate_chunk_id(self) -> int:
        ChunkList.generated_chunk_id += 1
        return ChunkList.generated_chunk_id

    def __getitem__(self, chunk_id: int):
        """
        Search a chunk by id.
        """
        return self.chunk_id_to_chunk_dict_map.get(chunk_id)

    def size(self) -> int:
        """
        Total number of chunks.
        """
        return len(self.chunk_id_to_chunk_dict_map)

    def get_chunk_memory_used(self, device):
        """
        The total memory of payload of all chunks on `device`.
        """
        mem_used = 0
        for _, chunk in self.chunk_id_to_chunk_dict_map.items():
            if (
                chunk.get_device() is not None
                and chunk.get_device().type == device.type
            ):
                mem_used += chunk.get_payload_space()
        return mem_used

    def max_chunk_size(self):
        max_size = 0
        for _, chunk in self.chunk_id_to_chunk_dict_map.items():
            max_size = max(chunk.capacity, max_size)
        return max_size

    def access_chunk(self, chunk_id: int, compute_device: torch.device):
        """
        Prepare the memory of chunk to `compute_device` with `chunk_id`.
        1. local mode
        In local mode, we need to move the chunk when it is on other devices.
            TODO(jiaruifang) Add async copy and record the lifecycle of chunks during
            the first iteration, so that we can prefetch the next chunk after sync
            the memcopy of the first chunk.
        2. distributed mode
        Use allgather to fetch chunks from other processes.
        """

        chunk = self.chunk_id_to_chunk_dict_map[chunk_id]

        # The moment of chunk accessing during warmup.
        mgr = PatrickStarManager()
        if mgr.is_warmup_training():
            cur_mem = mgr.get_cur_mom()
            chunk.append_moment(cur_mem, compute_device)

        chunk_status = chunk.get_status()

        payload_space = chunk.get_chunk_space()

        # If chunk was released, we need to reallocate it.
        # In distributed mode, we need a global payload.
        if chunk_status == PSChunkStatus.RELEASED:
            logger.debug(
                f"rank {get_rank()} access_chunk chunk {chunk_id}, "
                f"need to allocate {payload_space} B memory on {compute_device}"
            )

            # TODO(jiaruifang) We need to prepare for the distributed environment.
            self.prepare_device(compute_device, payload_space)
            chunk.allocate_payload(compute_device)
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
        """
        if self._time_profile:
            global_timer.my_timer.start_profile("CHUNK_LIST_prepare_device")

        mgr = PatrickStarManager()
        ava_chunk_mem_size = mgr.available_chunk_mem(target_device.type)
        free_chunk_mem_size = mgr.free_chunk_mem(target_device.type)

        logger.debug(
            f"prepare_target: device {target_device} need_bytes {need_bytes / 1e6} MB, "
            f"ava_chunk_mem_size {ava_chunk_mem_size / 1e6} MB, "
            f"free_chunk_mem_size {free_chunk_mem_size / 1e6} MB."
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

        extra_need_bytes = need_bytes - free_chunk_mem_size

        logger.debug(
            f"{target_device} (ava_chunk_mem_size {ava_chunk_mem_size / 1e6} MB) "
            f"now free_chunk_mem_size size {free_chunk_mem_size / 1e6} MB, "
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
            torch.device("cpu")
            if target_device.type == "cuda"
            else torch.device(f"cuda:{self.local_rank}")
        )

        # Move the chunk to new device. If there are not enough space on the new device, abort.
        for idx in moved_list:
            self.chunk_move(idx, new_device)

        if self._time_profile:
            global_timer.my_timer.finish_profile("CHUNK_LIST_prepare_device")

    def make_room(self, offload_size_in_bytes, target_device):
        """
        Move `offload_size_in_bytes` size of chunks away from `target_device`.
        Can not move chunk of status `COMPUTE`.
        """
        if self._time_profile:
            global_timer.my_timer.start_profile("CHUNK_LIST_make_room")

        moved_list = self._chunk_to_move_out_for_room_making(
            offload_size_in_bytes, target_device
        )

        new_device = (
            torch.device("cpu")
            if target_device.type == "cuda"
            else torch.device(f"cuda:{self.local_rank}")
        )

        for idx in moved_list:
            self.chunk_move(idx, new_device)

        if self._time_profile:
            global_timer.my_timer.finish_profile("CHUNK_LIST_make_room")

    def chunk_move(self, chunk_id: int, device: torch.device):
        """
        Move chunk of id `chunk_id` to `device`.
        NOTE(): Please make sure `device` has enough free_chunk_mem before.
        """
        if self._time_profile:
            global_timer.my_timer.start_profile("CHUNK_LIST_chunk_move")

        chunk = self.chunk_id_to_chunk_dict_map[chunk_id]

        mgr = PatrickStarManager()
        free_chunk_mem_size = mgr.free_chunk_mem(device.type)

        chunk_mem_size = chunk.get_payload_space()
        if free_chunk_mem_size < chunk_mem_size:
            raise RuntimeError(
                f"chunk move failed. {device} has not {chunk_mem_size / 1e6} MB memory space. "
                f"Free space is {free_chunk_mem_size / 1e6} MB. "
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
        chunk_type: ChunkListType = ChunkListType.UNDEF,
    ):
        """
        Create a chunk without initializing its memory.
        Return the index of the chunk in the comm group:
            (comm_group_idx, comm_group_offset)
        """
        if chunk_id in self.chunk_id_to_chunk_dict_map:
            raise RuntimeError(
                f"chunk list new chunk with chunk_id {chunk_id} already existed"
            )
        self.chunk_id_to_chunk_dict_map[chunk_id] = Chunk(
            capacity=chunk_size,
            data_type=data_type,
            chunk_id=chunk_id,
            local_rank=self.local_rank,
            is_dummy=is_dummy,
        )
        world_size = get_world_size()
        global_rank = get_rank()
        self.chunk_type_to_id_list_map[chunk_type].append(chunk_id)
        if profiler.started():
            profiler.chunk_life_cycle[chunk_id] = {"type": chunk_type, "life_cycle": []}
        tmp_chunk_list_len = len(self.chunk_type_to_id_list_map[chunk_type])
        comm_group_offset = (tmp_chunk_list_len - 1) % world_size
        comm_group_idx = (tmp_chunk_list_len - 1) // world_size
        logger.debug(
            f"global_rank {global_rank}, allocate with new chunk chunk_id {chunk_id} size {chunk_size} "
            f"data_type {data_type} comm group ({comm_group_idx}, {comm_group_offset}, {chunk_type})"
        )
        return comm_group_idx, comm_group_offset

    def is_empty(self, chunk_type: ChunkListType):
        return len(self.chunk_type_to_id_list_map[chunk_type]) == 0

    def last_chunk_id(self, chunk_type: ChunkListType):
        if self.is_empty(chunk_type):
            raise RuntimeError(
                f"Call last_chunk_id on an empty {chunk_type} chunk list"
            )
        return self.chunk_type_to_id_list_map[chunk_type][-1]

    def generate_chunk(self):
        for chunk_id, chunk in self.chunk_id_to_chunk_dict_map.items():
            yield chunk_id, chunk

    def _chunk_to_move_out_for_room_making(
        self, size_in_bytes: int, target_device: torch.device
    ) -> List:
        """
        Find the chunks to move for making `size_in_bytes` of room on `target_device`.
        Return a list of chunk_ids.
        """
        still_need_bytes = size_in_bytes
        moved_bytes = 0
        moved_list = []

        # TODO(jiaruifang) Now we are using a greedy method to find the chunk to move.
        # Find a better way with the lifecycle of the chunk.

        movable_chunk_info = []

        q = PriorityQueue()
        for chunk_id, chunk in self.chunk_id_to_chunk_dict_map.items():
            if (
                chunk.get_device() is not None
                and chunk.get_device().type == target_device.type
                and chunk.get_status() != PSChunkStatus.COMPUTE
                and chunk.is_pin() is False
            ):
                # The next moment when this chunk was accessed.
                next_mom = chunk.next_accessed_mom(target_device)
                # Order by `next_mom`s, from large to small
                # and by chunk_ids if `next_mom` are the same (only happens during warmup).
                q.put((-next_mom, chunk_id))
                movable_chunk_info.append(f"{next_mom}_{chunk_id}")
            # TODO(jiaruifang) Do not release `FREE` chunks immediately for reuse.
            # assert chunk.get_status() != PSChunkStatus.FREE
        while not q.empty():
            next_mom, chunk_id = q.get()
            moved_bytes += self.chunk_id_to_chunk_dict_map[chunk_id].get_payload_space()
            moved_list.append(chunk_id)
            if moved_bytes >= still_need_bytes:
                break

        mgr = PatrickStarManager()
        logger.info(
            f"**** EVICT INFO(next_mom, chunk_id) {target_device}: "
            f"cur_mom {mgr.get_cur_mom()} movable_chunk_info {movable_chunk_info}, "
            f"real moved_list {moved_list}"
        )
        # Raise error when failed to make enough room.
        if moved_bytes < still_need_bytes:
            raise RuntimeError(
                f"device {target_device} still needs {still_need_bytes / 1e6} MB, "
                f"but there is not enough space on it, only {moved_bytes / 1e6} MB available. "
                f"chunk mem used memory on {target_device} is "
                f"{self.get_chunk_memory_used(target_device) / 1e6} MB"
            )

        return moved_list

    def update_status(self, chunk_id, old_status, new_status):
        self.chunk_id_to_chunk_dict_map[chunk_id].update_status(old_status, new_status)

    def display_access_info(self):
        logger.debug("----------- SHOW ACCESS INFO -----------")
        for chunk_id in self.chunk_type_to_id_list_map[ChunkListType.PARAM_FP16]:
            chunk = self.chunk_id_to_chunk_dict_map[chunk_id]
            logger.debug(f"\t {chunk_id} cpu_access_moments {chunk.cpu_access_moments}")
            logger.debug(f"\t {chunk_id} gpu_access_moments {chunk.gpu_access_moments}")
