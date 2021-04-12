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

from chunk import Chunk
import sys
from manager import HybridPSManager
import logging
import torch
from const import PSChunkStatus
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

    def new_chunk(self, chunk_size: int) -> int:
        """
        新建一个chunk，返回它的id
        只有没有find_available_chunk失败才调用new_chunk
        """
        chunk_id = self.id
        self.chunk_id_to_chunk_dict[chunk_id] = Chunk(capacity=chunk_size,
                                                      chunk_id=chunk_id)
        self.id = self.id + 1
        return chunk_id, self.chunk_id_to_chunk_dict[chunk_id]

    def delete_chunk(self, chunk_id: int):
        """
        删除chunk_id对应的chunk
        """
        if chunk_id in self.chunk_id_to_chunk_dict:
            del self.chunk_id_to_chunk_dict[chunk_id]

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

    def allocate(self, size: int, tensor_id: id) -> (int, torch.Tensor):
        """
        找到chunk_list中可以分配size大小数据的chunk，如果没有则新分配一个
        返回chunk_id
        """
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            ret = chunk.try_allocate(size, tensor_id)
            if ret is not None:
                logging.debug(f'allocate with old chunk {chunk_id}')
                return chunk_id, ret
        # need allocate a new chunk
        chunk_id, chunk = self.new_chunk(max(size, self.default_chunk_size))
        ret = chunk.try_allocate(size, tensor_id)
        assert ret is not None
        logging.debug(f'allocate with new chunk {chunk_id}')
        return chunk_id, ret

    def __getitem__(self, chunk_id: int):
        """
        索引一个chunk
        """
        return self.chunk_id_to_chunk_dict[chunk_id]

    def size(self) -> int:
        """
        返回chunk的个数
        """
        return len(self.chunk_id_to_chunk_dict)

    def generate(self) -> (int, Chunk):
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            yield chunk_id, chunk

    def chunk_to_move_out_for_room_making(self,
                                          size: int,
                                          target_device: torch.device) -> List:
        """
        为target device腾出size大小，找出需要移动出哪些chunk
        返回一个chunk_id list
        """
        # 先找到可以free chunk
        free_chunk_id_list = []
        freed_size = 0
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk.device == target_device and chunk.get_status(
            ) == PSChunkStatus.FREE:
                free_chunk_id_list.append(chunk_id)
                freed_size += chunk.capacity

        for idx in free_chunk_id_list:
            self.delete_chunk(idx)

        if freed_size >= size:
            return

        # 如果还没有腾出足够的空间，则需要moved out hold状态的chunk
        still_need_size = size - freed_size
        moved_size = 0
        moved_list = []
        for chunk_id, chunk in self.chunk_id_to_chunk_dict.items():
            if chunk.device == target_device and chunk.get_status(
            ) == PSChunkStatus.HOLD:
                moved_size += chunk.capacity
                moved_list.append(chunk_id)

                if moved_size >= still_need_size:
                    break

        # 无法腾出足够空间，抛出异常
        if moved_size < still_need_size:
            for id, chunk in self.generate():
                chunk.visit()
            print(
                f"still need {still_need_size}, but device {target_device} has not enough space for item."
            )
            raise RuntimeError

        return moved_list


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    manager = HybridPSManager()
    manager.reset([32, 32], [1024])

    chunk_list = ChunkList(default_chunk_size=128)
    # 之前分配的chunk中尝试分配10空间
    chunk_id = chunk_list.least_used_chunk()
    assert chunk_id == 0

    chunk_id, tensor = chunk_list.allocate(10, 0)
    assert chunk_id == 0

    chunk_id, tensor = chunk_list.allocate(100, 1)
    assert chunk_id == 0
    chunk_id = chunk_list.least_used_chunk()
    chunk_list[chunk_id].visit()

    chunk_id, tensor = chunk_list.allocate(100, 2)
    assert (chunk_id == 1)

    chunk_id = chunk_list.least_used_chunk()
    assert chunk_id == 1
    # 再分配一个chunk
    # chunk_list.new_chunk(128)