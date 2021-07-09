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

from .chunk_data import Chunk
from .const import PSChunkStatus, AccessType, PSTensorStatus
from .helper import getsizeof

import sys
import logging
import torch
from typing import List
import time
from queue import PriorityQueue
import gc
import psutil

import patrickstar.utils.global_timer as global_timer
from patrickstar.utils.memory_monitor import see_memory_usage, get_sys_memory_used
from patrickstar.utils import logger, log_dist
from patrickstar.manager import PatrickStarManager
from patrickstar.deepspeed_helper.global_vars import get_args


class ChunkList(object):
    """
    管理一个chunk链表
    """
    def __init__(self, rank: int = 0):
        self.chunk_id_to_chunk_dict: dict[int, Chunk] = {}
        self._time_profile = True
        # TODO单GPU不能启动太多stream
        self.copy_stream = torch.cuda.Stream()
        self.moments_cnt_of_iteration = None

    def __getitem__(self, chunk_id: int):
        """
        索引一个chunk
        """
        return self.chunk_id_to_chunk_dict.get(chunk_id)

    def size(self) -> int:
        """
        返回chunk的个数
        """
        return len(self.chunk_id_to_chunk_dict)

    def get_chunk_memory_used(self, device):
        """
        获得ChunkList中所有Chunk的payload占用的内存
        """
        mem_used = 0
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk.get_device() is not None and chunk.get_device(
            ).type == device.type:
                mem_used += chunk.get_payload_space()
        return mem_used

    def max_chunk_size(self):
        max_size = 0
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
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
        chunk = self.chunk_id_to_chunk_dict[chunk_id]
        chunk_status = chunk.get_status()

        payload_space = chunk.get_chunk_space()

        # 如果chunk的内存释放了，需要将它分配出来
        # 分布式情况，需要分配一个全局的payload
        if chunk_status == PSChunkStatus.RELEASED:
            logger.debug(
                f'rank {torch.distributed.get_rank()} access_chunk chunk {chunk_id}, need to allocate {payload_space} B memory on {compute_device}'
            )

            # TODO 在分布式环境应该准备
            self.prepare_device(compute_device, payload_space)
            chunk.allocate_payload(compute_device)
            return
        elif chunk.get_device().type != compute_device.type:
            self.prepare_device(compute_device, payload_space)
            chunk.move(compute_device, self.copy_stream)
            assert chunk.get_device(
            ).type == compute_device.type, f"chunk device {chunk.get_device()} compute device {compute_device}"
            return
        else:
            # 目标chunk已经在计算设备上了
            logging.debug(
                f'access_chunk chunk {chunk_id} directly on {compute_device}')

    def prepare_device(self, target_device: torch.device, need_bytes: int):
        """
        让target device做腾出need_bytes大小
        如果空间不足，需要在目标设备上释放或者移动出一些chunk。
        """
        if self._time_profile:
            global_timer.my_timer.start_profile('CHUNK_LIST_prepare_device')

        ps_manager = PatrickStarManager()
        ava_chunk_mem_size = ps_manager.available_chunk_mem(target_device.type)
        free_chunk_mem_size = ps_manager.free_chunk_mem(target_device.type)

        logger.debug(
            f'prepare_target: device {target_device} need_bytes {need_bytes/1e6} MB, ava_chunk_mem_size {ava_chunk_mem_size/1e6} MB, free_chunk_mem_size {free_chunk_mem_size/1e6} MB.'
        )

        args = get_args()

        # TODO(jiaruifang) 无法分配的情况
        # 这个条件尚不充分，应该是如果cpu和gpu的的free_chunk都不足存放need bytes则放弃
        if ava_chunk_mem_size < need_bytes:
            logger.error(
                f"{target_device} has not enough space for {need_bytes} elements"
            )
            # TODO(jiaruifang)可以爆表时候再释放
            raise RuntimeError(
                f"{target_device} has not enough space for {need_bytes/1e6} MB. Device used Chunk Memory is {self.get_chunk_memory_used(target_device)/1e6} MB. Avaibale Chunk Memory is {ava_chunk_mem_size/1e6} MB"
            )

        extra_need_bytes = need_bytes - free_chunk_mem_size

        logger.debug(
            f'{target_device} (ava_chunk_mem_size {ava_chunk_mem_size/1e6} MB) now free_chunk_mem_size size {free_chunk_mem_size/1e6} MB, needs {need_bytes/1e6} MB'
        )
        # 不需要新分配
        if extra_need_bytes <= 0:
            if self._time_profile:
                global_timer.my_timer.finish_profile(
                    'CHUNK_LIST_prepare_device')
            return

        logger.debug(
            f'the device {target_device} has no enough free chunk memory, required size is {extra_need_bytes} bytes'
        )
        # 需要在target_device上腾出空间
        moved_list = self._chunk_to_move_out_for_room_making(
            extra_need_bytes, target_device)

        # TODO(jiaruifang) 这里默认新设备上有足够空间，强制把Chunk塞给新设备，可能会突破新设备上的ava_chunk_mem的上线，引起bug
        new_device = torch.device(
            'cpu') if target_device.type == 'cuda' else torch.device(
                f'cuda:{args.local_rank}')

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

        args = get_args()
        # 需要在target_device上腾出空间
        moved_list = self._chunk_to_move_out_for_room_making(
            offload_size_in_bytes, target_device)

        # TODO(jiaruifang)只考虑单卡情况，新设备只有gpu和cpu
        new_device = torch.device(
            'cpu') if target_device.type == 'cuda' else torch.device(
                f'cuda:{args.local_rank}')

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

        chunk = self.chunk_id_to_chunk_dict[chunk_id]

        ps_manager = PatrickStarManager()
        ava_chunk_mem_size = ps_manager.available_chunk_mem(device.type)
        free_chunk_mem_size = ps_manager.free_chunk_mem(device.type)

        chunk_mem_size = chunk.get_payload_space()
        if free_chunk_mem_size < chunk_mem_size:
            raise RuntimeError(
                f"chunk move failed. {device} has not {chunk_mem_size} MB memory space. Free space is {free_chunk_mem_size/1e6} MB. This is because the overall memory of CPU and GPU is not enough for the model."
            )
        if chunk.get_device() != device:
            logging.log(
                logging.DEBUG,
                f'move chunk {chunk_id} from {chunk.get_device()} to {device}')
            chunk.move(device, self.copy_stream)

        if self._time_profile:
            global_timer.my_timer.finish_profile('CHUNK_LIST_chunk_move')

    def new_chunk(self,
                  chunk_id: int,
                  chunk_size: int,
                  data_type: torch.dtype,
                  is_dummy: bool = False) -> int:
        """
        新建一个chunk，并未初始化内存
        """
        args = get_args()
        if chunk_id in self.chunk_id_to_chunk_dict:
            raise RuntimeError(
                f"chunk list new chunk with chunk_id {chunk_id} already existed"
            )
        self.chunk_id_to_chunk_dict[chunk_id] = Chunk(capacity=chunk_size,
                                                      data_type=data_type,
                                                      chunk_id=chunk_id,
                                                      rank=args.local_rank,
                                                      is_dummy=is_dummy)
        logging.debug(
            f'allocate with new chunk chunk_id {chunk_id} size {chunk_size} data_type {data_type}'
        )

    def least_used_chunk(self) -> int:
        """"
        返回最近被touch过的chunk
        """
        max_value = float('-inf')
        pos = 0
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk.get_timestamp() > max_value:
                max_value = chunk.get_timestamp()
                pos = chunk_id

        logging.debug(f'least_used_chunk found chunk id {pos}')
        return pos

    def generate_chunk(self) -> (int, Chunk):
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
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

        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            # TODO(jiaruifang) 耗时
            status = chunk.get_status()
            self._delete_chunk(chunk)

    def get_next_access_moment(self, chunk: Chunk,
                               target_device: torch.device):
        """
        找到chunk在本设备上下一次被访问的moment
        """
        # TODO还没加入统计信息
        return 0

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
        # 不是最优策略？应该按照访问顺序。
        # 找到lifecycle被需要最晚的chunk换出
        Q = PriorityQueue()
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk.get_device() is not None and chunk.get_device(
            ).type == target_device.type and chunk.get_status(
            ) != PSChunkStatus.COMPUTE and chunk.is_pin() is False:
                # 本设备下一次被需要的时刻？本设备下一次不被需要的时刻
                # 如果target_device 是cuda，
                next_mom = 0  #self.get_next_access_moment(chunk, target_device)
                # 按照next_mom从大到小排序，如果相同则按照chunk_id排序（只在预热阶段出现）
                Q.put((-next_mom, chunk_id))
            # TODO(jiaruifang)不立刻释放FREE chunk，而是让它参与复用
            # assert chunk.get_status() != PSChunkStatus.FREE
        while not Q.empty():
            next_mom, chunk_id = Q.get()
            moved_bytes += self.chunk_id_to_chunk_dict[
                chunk_id].get_payload_space()
            moved_list.append(chunk_id)
            if moved_bytes >= still_need_bytes:
                break

        # 无法腾出足够空间，抛出异常
        if moved_bytes < still_need_bytes:
            self.visit()
            raise RuntimeError(
                f"device {target_device} still needs {still_need_bytes/1e6} MB, but  it has not enough space, only {moved_bytes/1e6} MB available. ChunkList used memory on {target_device} is {self.get_chunk_memory_used(target_device)/1e6} MB"
            )

        return moved_list

    def update_status(self, chunk_id, old_status, new_status):
        self.chunk_id_to_chunk_dict[chunk_id].update_status(
            old_status, new_status)

    def visit(self):
        logging.info('* chunk list visit results:')
        logging.info('** chunk_id, device, size(B), ' 'type, device, status')
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            logging.info(
                f'** {chunk_id}, {chunk.get_device()}, {chunk.get_chunk_space()}, '
                f'{chunk.data_type}, {chunk.get_device()}, {chunk.get_status()}'
            )
