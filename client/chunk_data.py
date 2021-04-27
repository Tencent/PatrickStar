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

import os
import torch
from .const import PSTensorStatus, PSChunkStatus, AccessType
from .helper import getsizeof
from .chunk_tensor_index import ChunkTensorIndex
from .helper import getsizeof

from typing import Dict
from manager import HybridPSManager

import datetime
import logging
import time

total_cpu_gpu_move_elapse = 0.0
total_cpu_gpu_move_time = 0
total_cpu_gpu_move_amount = 0


class Chunk(object):
    def __init__(self, capacity: int, data_type: torch.dtype, chunk_id: int):
        """
        Chunk是数据迁移的最小单位，
        它用一段连续的内存来存储张量
        删除tensor，只需要将tensor的status设置为free
        """
        self.pid = os.getpid()
        self.chunk_id = chunk_id
        self.capacity = capacity
        self.data_type = data_type

        # 由调度决定分配在CPU还是GPU上
        self.ps_manager = HybridPSManager()
        if self.ps_manager.is_init() is False:
            raise "init Manager first before init a Chunk"

        self.cuda_idx = torch.cuda.current_device()

        self.device = self.ps_manager.schedule(capacity * getsizeof(data_type),
                                               self.cuda_idx)
        if self.device.type == 'cuda' and self.device.index > torch.cuda.device_count(
        ):
            logging.log(
                logging.WARNING,
                "When init a Chunk, the assigned cuda device idx is larger than available cuda count"
            )
            logging.log(logging.WARNING, "Set cuda index to 0")
            self.device = torch.cuda.current_device()

        self.payload = torch.zeros(capacity,
                                   dtype=self.data_type,
                                   device=self.device)
        self.ps_manager.add(self.device.type, self.device.index,
                            capacity * getsizeof(data_type))

        self.timestamp = datetime.datetime.now().timestamp()

    def get_size(self):
        return getsizeof(self.data_type) * self.capacity

    def touch(self):
        self.timestamp = datetime.datetime.now().timestamp()

    def get_timestamp(self):
        return self.timestamp

    def try_allocate(self, param: torch.nn.Parameter, access_type: AccessType,
                     chunk_tensor_index: ChunkTensorIndex) -> torch.Tensor:
        """
        在chunk的连续payload中找到一个满足param ps_data_size大小的碎片
        采用贪心算法，因为考虑NN参数的分配一般是连续分配，连续释放，没必要设计更复杂的算法
        """
        numel = param.ps_shape.numel()

        prev_end = 0
        for info in chunk_tensor_index.generate_tensor_info_in_order(
                self.chunk_id):
            status = info.status()
            if status == PSTensorStatus.FREE:
                continue
            start = info.start_offset
            gap = start - prev_end
            if gap >= numel:
                dest = self.allocate(prev_end, numel, param, access_type,
                                     chunk_tensor_index)
                return dest
            prev_end = start + info.numel

        if self.capacity - prev_end >= numel:
            dest = self.allocate(prev_end, numel, param, access_type,
                                 chunk_tensor_index)
            return dest
        return None

    def allocate(self, offset: int, numel: int, param: torch.nn.Parameter,
                 access_type: AccessType,
                 chunk_tensor_index: ChunkTensorIndex):
        """
        分配大小为numel的连续存储空间
        @params
        offset: 在chunk中的偏移
        numel: 分配tensor的元素个数
        access_type: DATA或者GRAD
        """
        dest = self.payload.narrow(0, offset, numel)
        # 复用内存要清零
        dest.zero_()
        # 在param中注册信息
        if access_type == AccessType.DATA:
            tensor_id = param.ps_data_id
        elif access_type == AccessType.GRAD:
            tensor_id = param.ps_grad_id
        chunk_tensor_index.add_tensor(self.chunk_id, tensor_id, offset, numel,
                                      param, access_type)
        self.touch()
        return dest

    def visit(self, chunk_tensor_index):
        """
        展示Chunk内所有tensor信息
        """
        logging.error(
            f'show chunk {self.chunk_id} capacity {self.capacity} dtype {self.data_type} device {self.device}'
        )
        for info in chunk_tensor_index.generate_tensor_info_in_order(
                self.chunk_id):
            assert info.chunk_id == self.chunk_id
            logging.error(
                f"** tensor: chunk_id {self.chunk_id}, device {self.device}, start {info.start_offset}, end {info.start_offset + info.numel}, tensor_id {info.tensor_id}, status {info.status()}"
            )

    def move(self,
             chunk_tensor_index: ChunkTensorIndex,
             target_device: torch.device,
             show_profile=False):
        """
        将这个chunk移动到target_device上。前提条件，target_device已经腾出足够的空间。
        """
        # logging.info(
        #     f'move chunk {self.chunk_id} numel {self.payload.numel()} from {self.device} to {target_device}'
        # )
        start_time = time.time()
        if self.device == target_device:
            return
        self.payload = self.payload.to(target_device)
        self.ps_manager.add(target_device.type, target_device.index,
                            self.capacity * getsizeof(self.data_type))
        self.ps_manager.delete(self.device.type, self.device.index,
                               self.capacity * getsizeof(self.data_type))
        # 将参数指针重新定位到新的设备上
        for info in chunk_tensor_index.generate_tensor_info_in_order(
                self.chunk_id):
            tensor_id = info.tensor_id
            start = info.start_offset
            numel = info.numel
            param = info.param
            access_type = info.access_type
            # if info.status() == PSTensorStatus.FREE:
            #     continue

            if access_type == AccessType.DATA:
                logging.debug(
                    f'chunk moves data tensor {tensor_id} to {target_device}')
                param.ps_data_tensor = self.payload.narrow(
                    0, start, numel).view(param.ps_shape)
                # 把data原来指向的内存释放
                param.data = param.ps_data_tensor
            elif access_type == AccessType.GRAD:
                logging.debug(
                    f'chunk moves grad tensor {tensor_id} to {target_device}')
                param.ps_grad_tensor = self.payload.narrow(
                    0, start, numel).view(param.ps_shape)

                # Note(jiaruifang)为了保证data和grad在统一设备上
                # 在移动ps_grad_tensor时，我们不能轻易改变grad指针
                # param.grad = param.ps_grad_tensor
                param.grad = None
            else:
                raise RuntimeError

        self.device = target_device
        self.touch()
        finish_time = time.time()
        elapse = finish_time - start_time

        if show_profile:
            global total_cpu_gpu_move_elapse
            global total_cpu_gpu_move_time
            global total_cpu_gpu_move_amount

            total_cpu_gpu_move_elapse += elapse
            total_cpu_gpu_move_time += 1
            total_cpu_gpu_move_amount += self.capacity * getsizeof(
                self.data_type)
            logging.info(
                f'CPU-GPU data move elapse {elapse} sec, total elapse {total_cpu_gpu_move_elapse} sec, total times {total_cpu_gpu_move_time}, total amount {total_cpu_gpu_move_amount/1e3} KB.'
            )


if __name__ == "__main__":
    manager = HybridPSManager()
    manager.reset([32, 32], [1024])

    from client import HybridPSClient
    client = HybridPSClient(0, torch.float, 20)
    chunk = Chunk(20, torch.float, 0)
    chunk.visit()

    param1 = torch.nn.Parameter(torch.zeros(10))
    client.register_param(param1)
    chunk.allocate(0, param1.numel(), param1, AccessType.DATA)

    param2 = torch.nn.Parameter(torch.zeros(10))
    client.register_param(param2)
    chunk.allocate(10, param2.numel(), param2, AccessType.DATA)

    param1.data_status = PSChunkStatus.FREE
    chunk.visit()
