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
from .const import PSTensorStatus, PSChunkStatus
from .helper import getsizeof
from .helper import getsizeof

from typing import Dict

import datetime
import logging
import time

from patrickstar.manager import PatrickStarManager
import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import logger


# chunk是否应该感知param？
class Chunk(object):
    def __init__(self,
                 capacity: int,
                 data_type: torch.dtype,
                 chunk_id: int,
                 rank: int = 0,
                 is_dummy: bool = False):
        """
        Chunk是数据迁移的最小单位，
        它用一段连续的内存来存储张量
        删除tensor，只需要将tensor的status设置为free
        这里把chunk设置为对是否分布式无感的，每个进程看到自己的chunk instance。
        """
        self.pid = os.getpid()
        self.chunk_id = chunk_id
        # payload numel 不等于 capacity, payload可能是None
        self.capacity = capacity
        self.data_type = data_type
        self.rank = rank
        self._is_dummy = is_dummy

        # 存储chunk管理tensor的状态数目
        self._status_dict = {
            PSTensorStatus.COMPUTE: 0,
            PSTensorStatus.HOLD: 0,
            PSTensorStatus.HOLD_AFTER_FWD: 0,
            PSTensorStatus.HOLD_AFTER_BWD: 0,
            PSTensorStatus.FREE: 0
        }

        self.payload = None
        self._time_profile = True

        self.gpu_access_moments = []
        self.cpu_access_moments = []
        self._pin_flag = False

    def append_moment(self, mom, compute_device):
        mgr = PatrickStarManager()
        assert mgr.is_warmup_training()

        access_moments = self.gpu_access_moments if compute_device.type == 'cuda' else self.cpu_access_moments
        if len(access_moments) > 0 and mom == access_moments[-1]:
            return
        else:
            access_moments.append(mom)

    def next_accessed_mom(self, compute_device):
        """
        非warmup时获得下一个访问时刻
        """
        mgr = PatrickStarManager()
        access_moments = self.gpu_access_moments if compute_device.type == 'cuda' else self.cpu_access_moments
        if mgr.is_nonwarmup_training():
            cur_mom = mgr.get_cur_mom()
            max_mom_small_than_cur = 0
            for i in access_moments:
                if i > cur_mom:
                    return i
                if i < cur_mom:
                    max_mom_small_than_cur = i
            return mgr.get_total_mom() + max_mom_small_than_cur
        else:
            return 0

    def display_access_mom_info(self):
        logger.info(
            f'\t {self.chunk_id} cpu_access_moments {self.cpu_access_moments}')
        logger.info(
            f'\t {self.chunk_id} gpu_access_moments {self.gpu_access_moments}')

    def is_dummy(self):
        return self._is_dummy

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
            global_timer.my_timer.start_profile('CHUNK_allocate_payload')

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
        ps_manager = PatrickStarManager()
        ps_manager.add(device.type, self.get_payload_space())

        self.touch()
        if self._time_profile:
            global_timer.my_timer.finish_profile('CHUNK_allocate_payload')

    def release_payload(self):
        """
        释放负载
        确保此时所有tensor都是free
        """
        ps_manager = PatrickStarManager()
        ps_manager.delete(self.get_device().type, self.get_payload_space())

        # 删除chunk的内存
        del self.payload
        self.payload = None

    def update_status(self, old_status, new_status):
        """
        更新chunk内tensor总体状态指标
        """
        self._status_dict[old_status] -= 1
        self._status_dict[new_status] += 1

    def status(self):
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

    def all_tensor_status(self, status):
        """
        判断所有tensor都是status状态
        """
        # TODO(jiaruifang) 可以优化一下
        tensor_num = 0
        for k, v in self._status_dict.items():
            if k != PSTensorStatus.FREE:
                tensor_num += v
        return self._status_dict[status] == tensor_num

    def touch(self):
        self.timestamp = datetime.datetime.now().timestamp()

    def get_timestamp(self):
        return self.timestamp

    def move(self, target_device: torch.device, copy_stream, is_async=False):
        """
        将这个chunk移动到target_device上。前提条件，target_device已经腾出足够的空间。
        """
        if self.get_device() is None:
            logging.warning(f"chunk move payload None to {target_device}")
            return
        if self.get_device() == target_device:
            return
        if self._time_profile:
            start_time = time.time()
            if target_device.type == 'cuda':
                global_timer.my_timer.start_profile('chunk_cpu_gpu_move')
            else:
                global_timer.my_timer.start_profile('chunk_gpu_cpu_move')
        src_device = self.get_device()
        ps_manager = PatrickStarManager()

        logging.debug(
            f'move chunk {self.chunk_id}, which has {self.payload.numel()/1e6} M {self.payload.dtype} elements, from {src_device} to {target_device}, used mem {ps_manager.used_chunk_mem(target_device.type)/1e6} MB'
        )

        #TODO(jiaruifang)异步
        ps_manager = PatrickStarManager()
        ps_manager.delete(self.get_device().type, self.get_payload_space())
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

        ps_manager.add(target_device.type, self.get_payload_space())

        if self._time_profile:
            if target_device.type == 'cuda':
                global_timer.my_timer.finish_profile('chunk_cpu_gpu_move')
                global_timer.data_move_cnter.update('chunk_cpu_gpu_move',
                                                    self.get_payload_space())
            elif target_device.type == 'cpu':
                global_timer.my_timer.finish_profile('chunk_gpu_cpu_move')
                global_timer.data_move_cnter.update('chunk_gpu_cpu_move',
                                                    self.get_payload_space())

    def get_device(self):
        if self.payload is not None:
            return self.payload.device
        else:
            return None
