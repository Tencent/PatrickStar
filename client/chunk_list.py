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
from .chunk_tensor_index import ChunkTensorIndex
from manager import HybridPSManager

import sys
import logging
import torch
from typing import List

import gc
import psutil
import utils.global_timer as global_timer
import time


def see_memory_usage(message, force=False):
    if not force:
        return
    if torch.distributed.is_initialized(
    ) and not torch.distributed.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    def get_tensors():
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data')
                                            and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                pass
                # print('A trivial exception occured: {}'.format(e))

    print('now let us see memory')
    for tensor in get_tensors():
        print(f'tensor shape {tensor.shape} {tensor.data_ptr()}')
    scale = 1  #1024 * 1024
    scale_name = "B"  #"MB"
    # Print message except when distributed but not rank 0
    logging.info(message)
    logging.info(
        f"MA {round(torch.cuda.memory_allocated() / scale,2 )} {scale_name} \
        Max_MA {round(torch.cuda.max_memory_allocated() / scale,2)} {scale_name} \
        CA {round(torch.cuda.memory_reserved() / scale,2)} {scale_name} \
        Max_CA {round(torch.cuda.max_memory_reserved() / scale)} {scale_name} "
    )

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logging.info(
        f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%'
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()


class ChunkList(object):
    """
    添加, O(1)
    删除, O(1)
    查找，遍历，查找有足够空闲碎片的chunk O(M, N), M tensor数量，N chunk数量
    索引，dict实现复杂度O(1)
    """
    def __init__(self, default_chunk_size: int):
        self.chunk_id_to_chunk_dict = {}
        self.default_chunk_size = default_chunk_size
        self.id = 0

    def new_chunk(self, chunk_size: int, data_type: torch.dtype) -> int:
        """
        新建一个chunk，返回它的id
        只有没有find_available_chunk失败才调用new_chunk
        """
        chunk_id = self.id
        self.chunk_id_to_chunk_dict[chunk_id] = Chunk(capacity=chunk_size,
                                                      data_type=data_type,
                                                      chunk_id=chunk_id)
        self.id = self.id + 1
        logging.debug(
            f'allocate with new chunk chunk_id {chunk_id} size {chunk_size} data_type {data_type}'
        )
        return chunk_id, self.chunk_id_to_chunk_dict[chunk_id]

    def delete_chunk(self, chunk_id: int, chunk_tensor_index: dict):
        """
        删除chunk_id对应的chunk，
        TODO(jiaruifang)还要删除chunk内的tensors
        """
        manager = HybridPSManager()
        if chunk_id in self.chunk_id_to_chunk_dict:
            start_time = time.time()
            chunk = self.chunk_id_to_chunk_dict[chunk_id]
            logging.debug(
                f'delete chunk id {chunk_id} size {chunk.capacity} type {chunk.data_type}'
            )
            manager.delete(chunk.device.type, chunk.device.index,
                           chunk.capacity * getsizeof(chunk.data_type))
            # 删除tensor的内存
            for info in chunk_tensor_index.generate_tensor_info_in_order(
                    chunk_id):
                param = info.param
                access_type = info.access_type
                if access_type == AccessType.DATA:
                    assert param.data_status == PSTensorStatus.FREE
                    param.ps_data_tensor = torch.zeros(
                        1, dtype=param.dtype, device=torch.device('cpu'))
                    param.data = param.ps_data_tensor
                elif access_type == AccessType.GRAD:
                    assert param.grad_status == PSTensorStatus.FREE
                    param.ps_grad_tensor = None
                    param.grad = None

            # 删除chunk的内存
            logging.debug(
                f'delete chunk id {chunk_id} payload of numel {self.chunk_id_to_chunk_dict[chunk_id].payload.numel()} on device {self.chunk_id_to_chunk_dict[chunk_id].device}'
            )

            # see_memory_usage('berfor delete payload', True)
            del self.chunk_id_to_chunk_dict[chunk_id]

            # see_memory_usage('after delete payload', True)
            # TODO(jiaruifang) delete tensor时候已经把chunk删除了
            chunk_tensor_index.delete_chunk_id(chunk_id)
            global_timer.memory_delete_elapse = time.time() - start_time

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

    def allocate(self, param: torch.nn.Parameter, access_type: AccessType,
                 chunk_tensor_index: ChunkTensorIndex) -> (int, torch.Tensor):
        """
        找到chunk_list中可以分配size大小数据的chunk，如果没有则新分配一个
        返回chunk_id
        """
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if param.dtype != chunk.data_type:
                continue
            ret = chunk.try_allocate(param, access_type, chunk_tensor_index)
            if ret is not None:
                logging.debug(f'allocate with old chunk {chunk_id}')
                return chunk_id, ret

        # need allocate a new chunk
        numel = param.ps_shape.numel()
        chunk_id, chunk = self.new_chunk(max(numel, self.default_chunk_size),
                                         param.dtype)
        ret = chunk.try_allocate(param, access_type, chunk_tensor_index)
        assert ret is not None
        logging.debug(
            f'no existing chunk can hold the tensor numel {numel}, new chunk {chunk_id}'
        )
        return chunk_id, ret

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

    def generate_chunk(self) -> (int, Chunk):
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            yield chunk_id, chunk

    def delete_free_chunks(self, chunk_tensor_index: ChunkTensorIndex):
        free_chunk_id_list = []
        freed_bytes = 0

        # 释放cpu和gpu上所有free chunk，统计目标设备上腾出的空间
        # NOTE(jiaruifang)这个循环很慢
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk_tensor_index.chunk_status(chunk_id) == PSChunkStatus.FREE:
                free_chunk_id_list.append(chunk_id)
                freed_bytes += chunk.get_size()

        # global_timer.delete_free_chunks_part1 += time.time() - start_time
        # 释放free chunks
        for idx in free_chunk_id_list:
            logging.debug(f'delete free chunk idx {idx}')
            self.delete_chunk(idx, chunk_tensor_index)

    def chunk_to_move_out_for_room_making(self, size_in_bytes: int,
                                          target_device: torch.device,
                                          chunk_tensor_index: ChunkTensorIndex
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
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk.device == target_device and chunk_tensor_index.chunk_status(
                    chunk_id) == PSChunkStatus.HOLD:
                moved_bytes += chunk.capacity * getsizeof(chunk.data_type)
                moved_list.append(chunk_id)

                if moved_bytes >= still_need_bytes:
                    break

            # TODO(jiaruifang)此时不应该有free状态的chunk，因为free在release时候完成了
            assert chunk_tensor_index.chunk_status(
                chunk_id) != PSChunkStatus.FREE

        # 无法腾出足够空间，抛出异常
        if moved_bytes < still_need_bytes:
            for chunk_id, chunk in self.generate_chunk():
                chunk.visit(chunk_tensor_index)
            logging.error(
                f"still need {still_need_bytes} bytes, but device {target_device} has not enough space for item."
            )
            raise RuntimeError

        return moved_list

    def show_stat(self):
        cuda_chunk_list = []
        cpu_chunk_list = []
        for chunk_id, chunk in self.generate_chunk():
            if chunk.device.type == 'cuda':
                cuda_chunk_list.append(chunk.capacity)
            elif chunk.device.type == 'cpu':
                cpu_chunk_list.append(chunk.capacity)
        logging.debug(f'cuda_chunk, {cuda_chunk_list}')
        logging.debug(f'cpu_chunk, {cpu_chunk_list}')


if __name__ == "__main__":
    from manager import HybridPSManager
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    manager = HybridPSManager()
    manager.reset([32, 32], [1024])

    data_type = torch.float
    chunk_list = ChunkList(default_chunk_size=128)
    # 之前分配的chunk中尝试分配10空间
    chunk_id = chunk_list.least_used_chunk()
    assert chunk_id == 0

    chunk_id, tensor = chunk_list.allocate(10, data_type, 0)
    assert chunk_id == 0

    chunk_id, tensor = chunk_list.allocate(100, data_type, 1)
    assert chunk_id == 0
    chunk_id = chunk_list.least_used_chunk()
    chunk_list[chunk_id].visit()

    chunk_id, tensor = chunk_list.allocate(100, data_type, 2)
    assert (chunk_id == 1)

    chunk_id = chunk_list.least_used_chunk()
    assert chunk_id == 1
    # 再分配一个chunk
    # chunk_list.new_chunk(128)
