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
from .const import PSChunkStatus, AccessType
from .helper import getsizeof
from .chunk_tensor_index import ChunkTensorIndex
from manager import HybridPSManager

import sys
import logging
import torch
from typing import List


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
            chunk = self.chunk_id_to_chunk_dict[chunk_id]
            logging.debug(
                f'delete chunk id {chunk_id} size {chunk.capacity} type {chunk.data_type}'
            )
            manager.delete(chunk.device.type, chunk.device.index,
                           chunk.capacity * getsizeof(chunk.data_type))
            # 此处删除chunk的payload，然是在payload上索引的tensor
            for info in chunk.tensor_info_list.generate_in_sorted_order():
                info.delete_tensor()
            del self.chunk_id_to_chunk_dict[chunk_id]
        chunk_tensor_index.delete_chunk_id(chunk_id)

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

    def allocate(self, param: torch.nn.Parameter,
                 access_type: AccessType) -> (int, torch.Tensor):
        """
        找到chunk_list中可以分配size大小数据的chunk，如果没有则新分配一个
        返回chunk_id
        """
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if param.dtype != chunk.data_type:
                continue
            ret = chunk.try_allocate(param, access_type)
            if ret is not None:
                logging.debug(f'allocate with old chunk {chunk_id}')
                return chunk_id, ret

        logging.debug(f'no existing chunk can hold the tensor size')
        # need allocate a new chunk
        numel = param.ps_shape.numel()
        chunk_id, chunk = self.new_chunk(max(numel, self.default_chunk_size),
                                         param.dtype)
        ret = chunk.try_allocate(param, access_type)
        assert ret is not None
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

    def generate(self) -> (int, Chunk):
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            yield chunk_id, chunk

    def delete_free_chunks(self, chunk_tensor_index: ChunkTensorIndex):
        free_chunk_id_list = []
        freed_bytes = 0

        # 释放cpu和gpu上所有free chunk，统计目标设备上腾出的空间
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk.get_status() == PSChunkStatus.FREE:
                free_chunk_id_list.append(chunk_id)
                freed_bytes += chunk.capacity * getsizeof(chunk.data_type)

        # 释放free chunks
        for idx in free_chunk_id_list:
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
        # 如果还没有腾出足够的空间，则需要moved out hold状态的chunk
        still_need_bytes = size_in_bytes
        moved_bytes = 0
        moved_list = []
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk.device == target_device and chunk.get_status(
            ) == PSChunkStatus.HOLD:
                moved_bytes += chunk.capacity * getsizeof(chunk.data_type)
                moved_list.append(chunk_id)

                if moved_bytes >= still_need_bytes:
                    break

        # 无法腾出足够空间，抛出异常
        if moved_bytes < still_need_bytes:
            for id, chunk in self.generate():
                chunk.visit()
            logging.error(
                f"still need {still_need_bytes} bytes, but device {target_device} has not enough space for item."
            )
            raise RuntimeError

        return moved_list

    def show_stat(self):
        cuda_chunk_list = []
        cpu_chunk_list = []
        for chunk_id, chunk in self.generate():
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
