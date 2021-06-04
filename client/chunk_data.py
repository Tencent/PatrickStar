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
    def __init__(self,
                 capacity: int,
                 data_type: torch.dtype,
                 chunk_id: int,
                 rank: int = 0):
        """
        Chunk是数据迁移的最小单位，
        它用一段连续的内存来存储张量
        删除tensor，只需要将tensor的status设置为free
        这里把chunk设置为对是否分布式无感的，每个进程看到自己的chunk instance。
        """
        # TODO(jiaruifang)标记本chunk是否在FWD和BWD被使用过，在FWD和BWD计算后被重置
        self.fwd_bwd_used = False
        self.pid = os.getpid()
        self.chunk_id = chunk_id
        # payload numel 不等于 capacity, payload可能是None
        self.capacity = capacity
        self.data_type = data_type
        self.rank = rank

        # 存储chunk管理tensor的状态数目
        self._status_dict = {
            PSTensorStatus.COMPUTE: 0,
            PSTensorStatus.HOLD: 0,
            PSTensorStatus.HOLD_AFTER_FWD: 0,
            PSTensorStatus.HOLD_AFTER_BWD: 0,
            PSTensorStatus.FREE: 0
        }

        self.ps_manager = HybridPSManager()
        if self.ps_manager.is_init() is False:
            raise "init Manager first before init a Chunk"

        self.payload = None
        self._time_profile = True

        self.access_moments = []

    def add_moment(self, mom):
        if len(self.access_moments) > 0 and self.access_moments[-1] == mom:
            return
        else:
            self.access_moments.append(mom)

    def get_chunk_space(self):
        """
        获取chunk的尺寸(Bytes)
        """
        return getsizeof(self.data_type) * self.capacity

    def get_payload_space(self):
        """
        获取payload的尺寸(Bytes)
        """
        if self.payload is None:
            return 0
        else:
            return getsizeof(self.payload.dtype) * self.payload.numel()

    def pin(self):
        self._pin_flag = True

    def unpin(self):
        self._pin_flag = False

    def is_pin(self):
        return self._pin_flag

    def allocate_payload(self, device):
        """
        为chunk分配payload，存储在compute_device上。
        NOTE()调用前保证compute有足够大的空间
        """
        if self._time_profile:
            start_time = time.time()

        payload_size = self.capacity
        if device.type == 'cpu':
            self.payload = torch.zeros(payload_size,
                                       dtype=self.data_type,
                                       device=device,
                                       pin_memory=True)
        else:
            self.payload = torch.zeros(payload_size,
                                       dtype=self.data_type,
                                       device=device)
        self.ps_manager.add(device.type, device.index,
                            self.get_payload_space())

        self.touch()
        if self._time_profile:
            global_timer.memory_allocate_elapse = time.time() - start_time

    def release_payload(self):
        """
        释放负载
        确保此时所有tensor都是free
        """
        # if self._time_profile:
        #     start_time = time.time()
        self.ps_manager.delete(self.get_device().type,
                               self.get_device().index,
                               self.get_payload_space())

        # 删除chunk的内存
        del self.payload
        self.payload = None

    def update_status(self, old_status, new_status):
        """
        更新chunk内tensor总体状态指标
        """
        self._status_dict[old_status] -= 1
        self._status_dict[new_status] += 1

    def show_life_cycle(self):
        logging.info(f'access_moments: {self.access_moments}')

    def get_status(self):
        """
        当没有payload时，状态是RELEASED
        有payload是，chunk状态由它所管理的tensor决定。
        """
        if self.payload is None:
            return PSChunkStatus.RELEASED

        # dist训练，需要强制把chunk固定在计算设备上
        if self._status_dict[PSTensorStatus.COMPUTE] > 0:
            return PSChunkStatus.COMPUTE
        elif self._status_dict[PSTensorStatus.HOLD] > 0:
            return PSChunkStatus.HOLD
        elif self._status_dict[PSTensorStatus.HOLD_AFTER_FWD] > 0:
            return PSChunkStatus.HOLD_AFTER_FWD
        elif self._status_dict[PSTensorStatus.HOLD_AFTER_BWD] > 0:
            return PSChunkStatus.HOLD_AFTER_BWD
        else:
            # uninit和free同等对待
            return PSChunkStatus.FREE

    def touch(self):
        self.timestamp = datetime.datetime.now().timestamp()

    def get_timestamp(self):
        return self.timestamp

    def move(self, target_device: torch.device, copy_stream, is_async=False):
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

        logging.debug(
            f'move chunk {self.chunk_id} numel {self.payload.numel()} from {self.get_device()} to {target_device}'
        )
        #TODO(jiaruifang)异步
        self.ps_manager.delete(self.get_device().type,
                               self.get_device().index,
                               self.get_payload_space())
        if is_async:
            if target_device.type == 'cpu':
                pinned_payload_cpu = torch.ones(self.payload.shape,
                                                dtype=self.payload.dtype,
                                                device='cpu:0',
                                                pin_memory=True)
                torch.cuda.synchronize()
                copy_stream.synchronize()
                with torch.cuda.stream(copy_stream):
                    pinned_payload_cpu.copy_(self.payload, non_blocking=True)
                copy_stream.synchronize()
                torch.cuda.synchronize()
                self.payload = pinned_payload_cpu
                # assert self.payload.pin_memory is True
            elif target_device.type == 'cuda':
                old_payload = self.payload
                self.payload = torch.ones(self.payload.shape,
                                          dtype=self.payload.dtype,
                                          device=f'cuda:{self.rank}')
                copy_stream.synchronize()
                # NOTE(jiaruifang) it is necessary
                torch.cuda.synchronize()
                copy_stream.synchronize()
                with torch.cuda.stream(copy_stream):
                    self.payload.copy_(old_payload, non_blocking=True)
                copy_stream.synchronize()
                torch.cuda.synchronize()

                assert (
                    torch.sum(old_payload.to(target_device) - self.payload)
                ) < 1e-3, f'old payload sum {torch.sum(old_payload)}, new payload sum {torch.sum(self.payload)}'
            else:
                raise RuntimeError
        else:
            self.payload = self.payload.to(target_device)

        self.ps_manager.add(target_device.type, target_device.index,
                            self.get_payload_space())
        self.touch()

        if self._time_profile:
            if target_device.type == 'cpu':
                global_timer.cpu_gpu_move_elapse += time.time() - start_time
                global_timer.cpu_gpu_move_times += 1
                global_timer.cpu_gpu_move_data_amount += self.get_payload_space(
                )
            elif target_device.type == 'cuda':
                global_timer.gpu_cpu_move_elapse += time.time() - start_time
                global_timer.gpu_cpu_move_times += 1
                global_timer.gpu_cpu_move_data_amount += self.get_payload_space(
                )

    def get_device(self):
        if self.payload is not None:
            return self.payload.device
        else:
            return None
