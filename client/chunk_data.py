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
from .helper import getsizeof

from typing import Dict
from manager import HybridPSManager

import datetime
import logging
import time
import utils.global_timer as global_timer


# chunk是否应该感知param？
class Chunk(object):
    def __init__(self, capacity: int, data_type: torch.dtype, chunk_id: int,
                 compute_device: torch.device):
        """
        Chunk是数据迁移的最小单位，
        它用一段连续的内存来存储张量
        删除tensor，只需要将tensor的status设置为free
        """
        self.pid = os.getpid()
        self.chunk_id = chunk_id
        self.capacity = capacity
        self.data_type = data_type
        self.compute_device = compute_device

        # 存储chunk管理tensor的状态数目
        self._status_dict = {
            PSTensorStatus.COMPUTE: 0,
            PSTensorStatus.HOLD: 0,
            PSTensorStatus.FREE: 0,
            PSTensorStatus.UNINIT: 0
        }

        self.ps_manager = HybridPSManager()
        if self.ps_manager.is_init() is False:
            raise "init Manager first before init a Chunk"

        self.payload = None
        self._time_profile = True

    def get_size(self):
        return getsizeof(self.data_type) * self.capacity

    def allocate_payload(self, device):
        """
        为chunk分配payload，存储在compute_device上。
        NOTE()调用前保证compute有足够大的空间
        """
        self.payload = torch.zeros(self.capacity,
                                   dtype=self.data_type,
                                   device=device)
        self.ps_manager.add(device.type, device.index, self.get_size())

        self.touch()

    def release_payload(self):
        """
        此时所有tensor都是free
        """
        if self._time_profile:
            start_time = time.time()

        self.ps_manager.delete(self.get_device().type,
                               self.get_device().index, self.get_size())

        # 删除chunk的内存
        del self.payload
        self.payload = None

        if self._time_profile:
            global_timer.memory_delete_elapse = time.time() - start_time

    def update_get_status(self, old_status, new_status):
        """
        更新chunk内tensor总体状态指标
        """
        self._status_dict[old_status] -= 1
        self._status_dict[new_status] += 1

    def get_status(self):
        """
        当没有payload时，状态是RELEASED
        有payload是，chunk状态由它所管理的tensor决定。
        """
        if self.payload is None:
            return PSChunkStatus.RELEASED

        if self._status_dict[PSTensorStatus.COMPUTE] > 0:
            return PSChunkStatus.COMPUTE
        elif self._status_dict[PSTensorStatus.HOLD] > 0:
            return PSChunkStatus.HOLD
        else:
            return PSChunkStatus.FREE

    def touch(self):
        self.timestamp = datetime.datetime.now().timestamp()

    def get_timestamp(self):
        return self.timestamp

    def move(self, target_device: torch.device):
        """
        将这个chunk移动到target_device上。前提条件，target_device已经腾出足够的空间。
        """
        if self._time_profile:
            start_time = time.time()
        if self.get_device() is None:
            logging.warning(f"chunk move payload None to {target_device}")
            return
        if self.get_device() == target_device:
            return

        # logging.info(
        #     f'move chunk {self.chunk_id} numel {self.payload.numel()} from {self.get_device()} to {target_device}'
        # )
        #TODO(jiaruifang)异步
        self.ps_manager.delete(self.get_device().type,
                               self.get_device().index, self.get_size())
        self.payload = self.payload.to(target_device)
        self.ps_manager.add(target_device.type, target_device.index,
                            self.get_size())
        self.touch()

        if self._time_profile:
            global_timer.cpu_gpu_move_elapse += time.time() - start_time
            global_timer.cpu_gpu_move_times += 1
            global_timer.cpu_gpu_move_data_amount += self.get_size()

    def get_device(self):
        if self.payload is not None:
            return self.payload.device
        else:
            return None

    # def try_allocate(self, param: torch.nn.Parameter, access_type: AccessType,
    #                  chunk_tensor_index: ChunkTensorIndex) -> torch.Tensor:
    #     """
    #     在chunk的连续payload中找到一个满足param ps_data_size大小的碎片
    #     采用贪心算法，因为考虑NN参数的分配一般是连续分配，连续释放，没必要设计更复杂的算法
    #     """
    #     numel = param.ps_shape.numel()

    #     prev_end = 0
    #     for info in chunk_tensor_index.generate_tensor_info_in_order(
    #             self.chunk_id):
    #         status = info.status()
    #         if status == PSTensorStatus.FREE:
    #             continue
    #         start = info.start_offset
    #         gap = start - prev_end
    #         if gap >= numel:
    #             dest = self.allocate(prev_end, numel, param, access_type,
    #                                  chunk_tensor_index)
    #             return dest
    #         prev_end = start + info.numel

    #     if self.capacity - prev_end >= numel:
    #         dest = self.allocate(prev_end, numel, param, access_type,
    #                              chunk_tensor_index)
    #         return dest
    #     return None

    # def allocate(self, offset: int, numel: int, param: torch.nn.Parameter,
    #              access_type: AccessType,
    #              chunk_tensor_index: ChunkTensorIndex) -> torch.Tensor:
    #     """
    #     分配大小为numel的tensor
    #     @params
    #     offset: 在chunk中的偏移
    #     numel: 分配tensor的元素个数
    #     access_type: DATA或者GRAD
    #     """
    #     dest = self.payload.narrow(0, offset, numel)
    #     # 复用内存要清零
    #     dest.zero_()
    #     # 在param中注册信息
    #     if access_type == AccessType.DATA:
    #         tensor_id = param.ps_data_id
    #     elif access_type == AccessType.GRAD:
    #         tensor_id = param.ps_grad_id
    #     if not hasattr(param, 'ps_name'):
    #         param.ps_name = None
    #     chunk_tensor_index.add_tensor(self.chunk_id, tensor_id, offset, numel,
    #                                   param, access_type)
    #     self.touch()
    #     return dest
