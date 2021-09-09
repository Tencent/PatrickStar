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

import patrickstar.utils.global_timer as global_timer
from patrickstar.core.const import ChunkListType
from patrickstar.manager import PatrickStarManager
from patrickstar.utils import logger, get_rank, get_world_size
from .chunk_data import Chunk
from .const import PSChunkStatus


class ChunkList(object):
    """
    管理一个chunk链表，
    需要区分四种chunk list，param fp16, param fp32, momentum, variance
    """
    generated_chunk_id = -1

    def __init__(self, local_rank: int):
        self.chunk_id_to_chunk_dict_map: dict[int, Chunk] = {}
        self.chunk_type_to_id_list_map: dict[ChunkListType, int] = {}
        for chunk_type in ChunkListType:
            self.chunk_type_to_id_list_map[chunk_type] = []

        self._time_profile = True
        # TODO(jiaruifang) 单GPU不能启动太多stream
        self.copy_stream = torch.cuda.Stream()
        self.moments_cnt_of_iteration = None
        self.local_rank = local_rank

    def chunk_ids_generator(self, chunk_list_type: ChunkListType):
        """
        生成chunk_list_type对应chunk的所有chunk_id
        """
        for chunk_id in self.chunk_type_to_id_list_map[chunk_list_type]:
            yield chunk_id

    def generate_chunk_id(self) -> int:
        ChunkList.generated_chunk_id += 1
        return ChunkList.generated_chunk_id

    def __getitem__(self, chunk_id: int):
        """
        索引一个chunk
        """
        return self.chunk_id_to_chunk_dict_map.get(chunk_id)

    def size(self) -> int:
        """
        返回chunk的个数
        """
        return len(self.chunk_id_to_chunk_dict_map)

    def get_chunk_memory_used(self, device):
        """
        获得ChunkList中所有Chunk的payload占用的内存
        """
        mem_used = 0
        for _, chunk in self.chunk_id_to_chunk_dict_map.items():
            if chunk.get_device() is not None and chunk.get_device(
            ).type == device.type:
                mem_used += chunk.get_payload_space()
        return mem_used

    def max_chunk_size(self):
        max_size = 0
        for _, chunk in self.chunk_id_to_chunk_dict_map.items():
            max_size = max(chunk.capacity, max_size)
        return max_size

    def access_chunk(self, chunk_id: int, compute_device: torch.device):
        """
        访问chunk_id，将chunk的内存准备到compute device上。
        1. local
        如果chunk在本进程其他设备上，需要移动。
            TODO(jiaruifang)异步移动，在第一次迭代统计chunk的声明周期。
            调用本函数时，先执行对chunk_id的同步操作。
            再发起对下一个chunk_id的预取。
        如果chunk的内存被释放，需要分配。
        2. distributed
        需要获取其他进程chunk，进行一次allgather获取一个完成的global chunk
        """

        chunk = self.chunk_id_to_chunk_dict_map[chunk_id]

        # 预热时注册chunk的访问时间
        mgr = PatrickStarManager()
        if mgr.is_warmup_training():
            cur_mem = mgr.get_cur_mom()
            chunk.append_moment(cur_mem, compute_device)

        chunk_status = chunk.get_status()

        payload_space = chunk.get_chunk_space()

        # 如果chunk的内存释放了，需要将它分配出来
        # 分布式情况，需要分配一个全局的payload
        if chunk_status == PSChunkStatus.RELEASED:
            logger.debug(
                f'rank {get_rank()} access_chunk chunk {chunk_id}, '
                f'need to allocate {payload_space} B memory on {compute_device}'
            )

            # TODO 在分布式环境应该准备
            self.prepare_device(compute_device, payload_space)
            chunk.allocate_payload(compute_device)
            return
        elif chunk.get_device().type != compute_device.type:
            self.prepare_device(compute_device, payload_space)
            chunk.move(compute_device, self.copy_stream)
            assert chunk.get_device().type == compute_device.type, (
                f"chunk device {chunk.get_device()} compute device {compute_device}")
            return
        else:
            # 目标chunk已经在计算设备上了
            logger.debug(
                f'access_chunk chunk {chunk_id} already on {compute_device}')

    def prepare_device(self, target_device: torch.device, need_bytes: int):
        """
        让target device做腾出need_bytes大小
        如果空间不足，需要在目标设备上释放或者移动出一些chunk。
        """
        if self._time_profile:
            global_timer.my_timer.start_profile('CHUNK_LIST_prepare_device')

        mgr = PatrickStarManager()
        ava_chunk_mem_size = mgr.available_chunk_mem(target_device.type)
        free_chunk_mem_size = mgr.free_chunk_mem(target_device.type)

        logger.debug(
            f'prepare_target: device {target_device} need_bytes {need_bytes / 1e6} MB, '
            f'ava_chunk_mem_size {ava_chunk_mem_size / 1e6} MB, '
            f'free_chunk_mem_size {free_chunk_mem_size / 1e6} MB.')

        # TODO(jiaruifang) 无法分配的情况
        # 这个条件尚不充分，应该是如果cpu和gpu的的free_chunk都不足存放need bytes则放弃
        if ava_chunk_mem_size < need_bytes:
            logger.error(
                f"{target_device} has not enough space for {need_bytes} elements"
            )
            # TODO(jiaruifang)可以爆表时候再释放
            raise RuntimeError(
                f'{target_device} has not enough space for {need_bytes / 1e6} MB. '
                f'Device used Chunk Memory is {self.get_chunk_memory_used(target_device) / 1e6} MB. '
                f'Avaibale Chunk Memory is {ava_chunk_mem_size / 1e6} MB')

        extra_need_bytes = need_bytes - free_chunk_mem_size

        logger.debug(
            f'{target_device} (ava_chunk_mem_size {ava_chunk_mem_size / 1e6} MB) '
            f'now free_chunk_mem_size size {free_chunk_mem_size / 1e6} MB, '
            f'needs {need_bytes / 1e6} MB')
        # 不需要新分配
        if extra_need_bytes <= 0:
            if self._time_profile:
                global_timer.my_timer.finish_profile(
                    'CHUNK_LIST_prepare_device')
            return

        logger.debug(
            f'the device {target_device} has no enough free chunk memory, '
            f'required size is {extra_need_bytes} bytes')
        # 需要在target_device上腾出空间
        moved_list = self._chunk_to_move_out_for_room_making(
            extra_need_bytes, target_device)

        # TODO(jiaruifang) 这里默认新设备上有足够空间，强制把Chunk塞给新设备，可能会突破新设备上的ava_chunk_mem的上线，引起bug
        new_device = torch.device(
            'cpu') if target_device.type == 'cuda' else torch.device(
                f'cuda:{self.local_rank}')

        # 把他们移动到新设备上，如果新设备上free_chunk空间不足，则放弃
        for idx in moved_list:
            self.chunk_move(idx, new_device)

        if self._time_profile:
            global_timer.my_timer.finish_profile('CHUNK_LIST_prepare_device')

    def make_room(self, offload_size_in_bytes, target_device):
        """
        让target_device移动出offload size大小的Chunk。
        不能移动compute状态的Chunk
        """
        if self._time_profile:
            global_timer.my_timer.start_profile('CHUNK_LIST_make_room')

        # 需要在target_device上腾出空间
        moved_list = self._chunk_to_move_out_for_room_making(
            offload_size_in_bytes, target_device)

        # TODO(jiaruifang)只考虑单卡情况，新设备只有gpu和cpu
        new_device = torch.device(
            'cpu') if target_device.type == 'cuda' else torch.device(
                f'cuda:{self.local_rank}')

        # 把他们移动到新设备上
        for idx in moved_list:
            self.chunk_move(idx, new_device)

        if self._time_profile:
            global_timer.my_timer.finish_profile('CHUNK_LIST_make_room')

    def chunk_move(self, chunk_id: int, device: torch.device):
        """
        将chunk_id的chunk移动到device上
        Note(): 必须保证device上有足够的free_chunk_mem
        """
        if self._time_profile:
            global_timer.my_timer.start_profile('CHUNK_LIST_chunk_move')

        chunk = self.chunk_id_to_chunk_dict_map[chunk_id]

        mgr = PatrickStarManager()
        free_chunk_mem_size = mgr.free_chunk_mem(device.type)

        chunk_mem_size = chunk.get_payload_space()
        if free_chunk_mem_size < chunk_mem_size:
            raise RuntimeError(
                f'chunk move failed. {device} has not {chunk_mem_size / 1e6} MB memory space. '
                f'Free space is {free_chunk_mem_size / 1e6} MB. '
                f'The reason may be that the overall memory of CPU and GPU is not enough for the model.'
            )
        if chunk.get_device() != device:
            logger.debug(
                f'move chunk {chunk_id} from {chunk.get_device()} to {device}')
            chunk.move(device, self.copy_stream)

        if self._time_profile:
            global_timer.my_timer.finish_profile('CHUNK_LIST_chunk_move')

    def new_chunk(self,
                  chunk_id: int,
                  chunk_size: int,
                  data_type: torch.dtype,
                  is_dummy: bool = False,
                  chunk_type: ChunkListType = ChunkListType.UNDEF):
        """
        新建一个chunk，并未初始化内存
        返回在通信组中的坐标，(comm_group_idx, comm_group_offset)
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
            is_dummy=is_dummy)
        world_size = get_world_size()
        global_rank = get_rank()
        self.chunk_type_to_id_list_map[chunk_type].append(chunk_id)
        tmp_chunk_list_len = len(self.chunk_type_to_id_list_map[chunk_type])
        comm_group_offset = (tmp_chunk_list_len - 1) % world_size
        comm_group_idx = (tmp_chunk_list_len - 1) // world_size
        logger.debug(
            f'global_rank {global_rank}, allocate with new chunk chunk_id {chunk_id} size {chunk_size} '
            f'data_type {data_type} comm group ({comm_group_idx}, {comm_group_offset}, {chunk_type})'
        )
        return comm_group_idx, comm_group_offset

    def is_empty(self, chunk_type: ChunkListType):
        return len(self.chunk_type_to_id_list_map[chunk_type]) == 0

    def last_chunk_id(self, chunk_type: ChunkListType):
        if self.is_empty(chunk_type):
            raise RuntimeError(
                f"Call last_chunk_id on an empty {chunk_type} chunk list")
        return self.chunk_type_to_id_list_map[chunk_type][-1]

    def generate_chunk(self):
        for chunk_id, chunk in self.chunk_id_to_chunk_dict_map.items():
            yield chunk_id, chunk

    def _delete_chunk(self, chunk: Chunk):
        """
        @depracated没有被调用
        删除chunk_id对应的chunk的payload
        Note(调用时chunk管理的tensor必须都是free的)
        """
        chunk.release_payload()

    def delete_free_chunks(self):
        """
        试图删除当前不被使用的chunk，即chunk内的tensor都是free状态的chunk
        """
        # 释放cpu和gpu上所有free chunk，统计目标设备上腾出的空间

        for _, chunk in self.chunk_id_to_chunk_dict_map.items():
            self._delete_chunk(chunk)

    def _chunk_to_move_out_for_room_making(self, size_in_bytes: int,
                                           target_device: torch.device
                                           ) -> List:
        """
        为target device腾出size大小，找出需要移动出哪些chunk
        先释放cpu，gpu的所有free
        返回一个chunk_id list
        """
        # 则需要将hold状态的chunk移出
        still_need_bytes = size_in_bytes
        moved_bytes = 0
        moved_list = []

        # TODO(jiaruifang)目前贪心地找到应该移动出去的chunk
        # 找到lifecycle被需要最晚的chunk换出

        movable_chunk_info = []

        q = PriorityQueue()
        for chunk_id, chunk in self.chunk_id_to_chunk_dict_map.items():
            if chunk.get_device() is not None and chunk.get_device(
            ).type == target_device.type and chunk.get_status(
            ) != PSChunkStatus.COMPUTE and chunk.is_pin() is False:
                # Chunk在本设备下一次被需要的时刻
                next_mom = chunk.next_accessed_mom(target_device)
                # 按照next_mom从大到小排序，如果相同则按照chunk_id排序（只在预热阶段出现）
                q.put((-next_mom, chunk_id))
                movable_chunk_info.append(f"{next_mom}_{chunk_id}")
            # TODO(jiaruifang)不立刻释放FREE chunk，而是让它参与复用
            # assert chunk.get_status() != PSChunkStatus.FREE
        while not q.empty():
            next_mom, chunk_id = q.get()
            moved_bytes += self.chunk_id_to_chunk_dict_map[
                chunk_id].get_payload_space()
            moved_list.append(chunk_id)
            if moved_bytes >= still_need_bytes:
                break

        mgr = PatrickStarManager()
        logger.info(
            f'**** EVICT INFO(next_mom, chunk_id) {target_device}: '
            f'cur_mom {mgr.get_cur_mom()} movable_chunk_info {movable_chunk_info}, '
            f'real moved_list {moved_list}')
        # 无法腾出足够空间，抛出异常
        if moved_bytes < still_need_bytes:
            raise RuntimeError(
                f'device {target_device} still needs {still_need_bytes / 1e6} MB, '
                f'but there is not enough space on it, only {moved_bytes / 1e6} MB available. '
                f'chunk mem used memory on {target_device} is '
                f'{self.get_chunk_memory_used(target_device) / 1e6} MB')

        return moved_list

    def update_status(self, chunk_id, old_status, new_status):
        self.chunk_id_to_chunk_dict_map[chunk_id].update_status(
            old_status, new_status)
