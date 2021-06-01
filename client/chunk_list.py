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
from .const import PSChunkStatus, AccessType, PSTensorStatus, PSChunkLocStatus
from .helper import getsizeof
from manager import HybridPSManager

import sys
import logging
import torch
from typing import List

import gc
import psutil
import utils.global_timer as global_timer
from utils.memory_monitor import see_memory_usage, get_memory_used
import time
from utils import logger, log_dist, debug_flag

from queue import PriorityQueue


class ChunkList(object):
    """
    管理一个chunk链表
    """
    def __init__(self, rank: int = 0):
        self.chunk_id_to_chunk_dict: dict[int, Chunk] = {}
        self._time_profile = True
        # TODO单GPU不能启动太多stream
        self.copy_stream = None if debug_flag else torch.cuda.Stream()
        self.moments_cnt_of_iteration = None
        self.rank = rank

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

    def max_chunk_size(self):
        max_size = 0
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            max_size = max(chunk.capacity, max_size)
        return max_size

    def _prepare_payload_from_remote_chunk(self, local_chunk_id,
                                           chunk_id_list):
        """
        为chunk_id_list中的chunk准备payload。
        其中本进程负责维护local_chunk_id的chunk payload
        远程的进程需要通过allgather通信获取payload
        结果是，每个进程chunk_id_list中对应的chunk payload的在本地
        """
        local_chunk_payload = self.chunk_id_to_chunk_dict[
            local_chunk_id].payload
        local_chunk_device = local_chunk_payload.device
        allgather_payload_buff = []

        for chunk_id in chunk_id_list:
            if chunk_id == local_chunk_id:
                allgather_payload_buff.append(local_chunk_payload)
            else:
                self.chunk_id_to_chunk_dict[chunk_id].allocate_payload(
                    local_chunk_device)
                allgather_payload_buff.append(
                    self.chunk_id_to_chunk_dict[chunk_id].payload)

        assert torch.distributed.is_initialized(
        ), "torch distributed is not initialized during allgather"

        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        log_dist(
            f'before allgather on rank {rank} buff {allgather_payload_buff}',
            [0, 1])
        handle = torch.distributed.all_gather(allgather_payload_buff,
                                              local_chunk_payload,
                                              async_op=False)
        log_dist(
            f'after allgather on rank {rank} buff {allgather_payload_buff}',
            [0, 1])
        # allgather_payload_buff的item要立刻释放，否则payload释放后还在allgather_payload_buff中保存
        allgather_payload_buff = []
        return handle

    def access_chunk_dist(self, local_chunk_id: int, chunk_id_list: List[int],
                          compute_device: torch.device):
        """
        在分布式训练中访问local_chunk_id对应的chunk，
        结果是把global_chunk_id管辖的chunk_id都填满
        """
        if self._time_profile:
            start_time = time.time()
        chunk = self.chunk_id_to_chunk_dict[local_chunk_id]
        chunk_status = chunk.get_status()
        local_space = chunk.get_chunk_space()

        world_size = torch.distributed.get_world_size()
        # 全局的payload需要的空间
        payload_space = local_space * world_size

        if chunk_status == PSChunkStatus.RELEASED:
            log_dist(
                f'access_chunk chunk {local_chunk_id}, need to allocate {payload_space} B memory on {compute_device}',
                [0, 1])
            self.prepare_device(compute_device, payload_space)

            chunk.allocate_payload(compute_device)
            # TODO(jiaruifang) 分配 n_proc - 1个chunk payload作为buffer
            self._prepare_payload_from_remote_chunk(local_chunk_id,
                                                    chunk_id_list)
            if self._time_profile:
                global_timer.access_chunk_elapse += time.time() - start_time
            return
        elif chunk.get_device().type != compute_device.type:
            log_dist(
                f'access_chunk chunk {local_chunk_id} prepare {payload_space} B memory on {compute_device}',
                [0, 1])
            self.prepare_device(compute_device, payload_space)
            chunk.move(compute_device, self.copy_stream)
            # TODO(jiaruifang) 分配 n_proc - 1个chunk payload作为buffer

            log_dist(
                f'move chunk {local_chunk_id} to {compute_device} payload {chunk.payload}',
                [0, 1])
            self._prepare_payload_from_remote_chunk(local_chunk_id,
                                                    chunk_id_list)

            assert chunk.get_device(
            ).type == compute_device.type, f"chunk device {chunk.get_device()} compute device {compute_device}"
            # TODO(jiaruifang) 分配 n_proc - 1个chunk payload作为buffer
            # 然后allgather

            if self._time_profile:
                global_timer.access_chunk_elapse += time.time() - start_time
            return
        else:
            log_dist(
                f'access_chunk chunk {local_chunk_id} directly on {compute_device}',
                [0, 1])

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
        if self._time_profile:
            start_time = time.time()

        chunk = self.chunk_id_to_chunk_dict[chunk_id]
        chunk_status = chunk.get_status()

        payload_space = chunk.get_chunk_space()

        # 如果chunk的内存释放了，需要将它分配出来
        # 分布式情况，需要分配一个全局的payload
        if chunk_status == PSChunkStatus.RELEASED:
            rank = torch.distributed.get_rank()
            logger.debug(
                f'access_chunk chunk {chunk_id}, need to allocate {payload_space} B memory on {compute_device}'
            )
            self.prepare_device(compute_device, payload_space)

            chunk.allocate_payload(compute_device)
            if self._time_profile:
                global_timer.access_chunk_elapse += time.time() - start_time
            return
        # 如果chunk目前所在的设备和计算设备不一致，
        # 光chunk的内存move是不够的，还需要param都move
        # 只有chunk状态是hold的会被移动，而hold状态的chunk中所有tensor都是hold或者free。
        # 这种tensor的内存都悬空
        elif chunk.get_device().type != compute_device.type:
            rank = torch.distributed.get_rank()
            self.prepare_device(compute_device, payload_space)
            chunk.move(compute_device, self.copy_stream)
            assert chunk.get_device(
            ).type == compute_device.type, f"chunk device {chunk.get_device()} compute device {compute_device}"
            if self._time_profile:
                global_timer.access_chunk_elapse += time.time() - start_time
            return
        else:
            logging.debug(
                f'access_chunk chunk {chunk_id} directly on {compute_device}')

    def prepare_device(self, target_device: torch.device, need_bytes: int):
        """
        让target device做分配need_bytes大小空间的准备
        如果空间不足，需要在目标设备上释放或者移动出一些chunk。
        """
        if self._time_profile:
            start_time = time.time()

        logging.log(
            logging.DEBUG,
            f'prepare_device target device {target_device} need size {need_bytes} bytes'
        )
        ps_manager = HybridPSManager()
        max_mem = ps_manager.max_mem(target_device.type, target_device.index)
        if max_mem < need_bytes:
            logging.log(
                logging.ERROR,
                f"{target_device} has not enough space for {need_bytes} elements"
            )
            # TODO(jiaruifang)可以爆表时候再释放
            raise RuntimeError(
                f"{target_device} has not enough space for {need_bytes} Bytes")

        available_size = ps_manager.available_mem(target_device.type,
                                                  target_device.index)

        # 当前系统可用内存，需要减去activation消耗
        # available_size = get_memory_used(target_device)

        extra_need_bytes = need_bytes - available_size

        logging.debug(
            f'{target_device} (max size {max_mem} B) now available size {available_size} B needs {need_bytes} B'
        )
        # 不需要新分配
        if extra_need_bytes <= 0:
            return

        logging.log(
            logging.DEBUG,
            f'the device {target_device} has no enough free space, extra size is {extra_need_bytes} bytes'
        )
        # 需要在target_device上腾出空间
        moved_list = self._chunk_to_move_out_for_room_making(
            extra_need_bytes, target_device)

        # TODO(jiaruifang)只考虑单卡情况，新设备只有gpu和cpu
        new_device = torch.device(
            'cpu') if target_device.type == 'cuda' else torch.device(
                f'cuda:{self.rank}')

        # 把他们移动到新设备上
        for idx in moved_list:
            self.chunk_move(idx, new_device)

        if self._time_profile:
            global_timer.client_prepare_device_elapse += time.time(
            ) - start_time

    def chunk_move(self, chunk_id: int, device: torch.device):
        """
        将chunk_id的chunk移动到device上
        """
        if self._time_profile:
            start_time = time.time()

        chunk = self.chunk_id_to_chunk_dict[chunk_id]
        if chunk.get_device() != device:
            logging.log(
                logging.DEBUG,
                f'move chunk {chunk_id} from {chunk.get_device()} to {device}')
            chunk.move(device, self.copy_stream)

        if self._time_profile:
            global_timer.chunk_move_elapse += time.time() - start_time

    def new_chunk(self, chunk_id: int, chunk_size: int,
                  data_type: torch.dtype) -> int:
        """
        新建一个chunk，并未初始化内存
        """
        if chunk_id in self.chunk_id_to_chunk_dict:
            raise RuntimeError(
                f"chunk list new chunk with chunk_id {chunk_id} already existed"
            )
        self.chunk_id_to_chunk_dict[chunk_id] = Chunk(capacity=chunk_size,
                                                      data_type=data_type,
                                                      chunk_id=chunk_id,
                                                      rank=self.rank)
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
            if self._time_profile:
                sub_start_time = time.time()
            # TODO(jiaruifang) 耗时
            status = chunk.get_status()
            if self._time_profile:
                global_timer.memory_delete_elapse += time.time(
                ) - sub_start_time

            if status == PSChunkStatus.FREE and chunk.location_status == PSChunkLocStatus.REMOTE:
                self._delete_chunk(chunk)

    def get_next_access_moment(self, chunk: Chunk,
                               target_device: torch.device):
        """
        找到chunk在本设备上下一次被访问的moment
        """
        # 还没加入统计信息
        timer = global_timer.IterationTimer()
        cur_moment = timer.moment()
        return 0
        # 预热阶段，返回值固定
        if timer.warmup:
            return 0

        # 非预热阶段
        target_device_type = target_device.type

        for mom in chunk.access_moments:
            # 找到在当前mom之后的时刻，且该时刻计算设备不是target device
            if mom >= cur_moment and timer.device_type(
                    mom) == target_device_type:
                return mom
        return self.moments_cnt_of_iteration + chunk.access_moments[0]

    def _chunk_to_move_out_for_room_making(self, size_in_bytes: int,
                                           target_device: torch.device
                                           ) -> List:
        """
        为target device腾出size大小，找出需要移动出哪些chunk
        先释放cpu，gpu的所有free
        返回一个chunk_id list
        """
        # 则需要将hold状态的chunk移出
        if self._time_profile:
            start_time = time.time()
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
            ) != PSChunkStatus.COMPUTE:
                # 本设备下一次被需要的时刻？本设备下一次不被需要的时刻
                # 如果target_device 是cuda，
                next_mom = 0  #self.get_next_access_moment(chunk, target_device)
                # 按照next_mom从大到小排序，如果相同则按照chunk_id排序（只在预热阶段出现）
                Q.put((-next_mom, chunk_id))
            # TODO(jiaruifang)不立刻释放FREE chunk，而是让它参与复用
            # assert chunk.get_status() != PSChunkStatus.FREE

        while Q:
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
                f"still need {still_need_bytes} bytes, but device {target_device} has not enough space for item."
            )

        if self._time_profile:
            global_timer.chunk_to_move_out_for_room_making_elapse += time.time(
            ) - start_time
        return moved_list

    def update_status(self, chunk_id, old_status, new_status):
        self.chunk_id_to_chunk_dict[chunk_id].update_status(
            old_status, new_status)

    def visit(self):
        ps_manager = HybridPSManager()
        ps_manager.visit()
        logging.info('* chunk list visit results:')
        logging.info('** chunk_id, device, size(B), ' 'type, device, status')
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            logging.info(
                f'** {chunk_id}, {chunk.get_device()}, {chunk.get_chunk_space()}, '
                f'{chunk.data_type}, {chunk.get_device()}, {chunk.get_status()}'
            )
            chunk.show_life_cycle()
