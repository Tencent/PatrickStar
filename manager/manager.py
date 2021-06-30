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

import torch
from torch.multiprocessing import Process, Manager
from utils.memory_monitor import get_sys_memory_used
import psutil
import logging as logger
from deepspeed_helper.global_vars import get_args


######### Global Scheduler ###########
class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Metronome():
    """节拍器"""
    def __init__(self):
        self._moment = 0
        self._total_moment = None

    def tiktac(self):
        self._moment += 1

    def moment(self):
        return self._moment

    def reset(self):
        self._total_moment = self._moment
        self._moment = 0

    def next_moment(self):
        assert self._total_moment is not None
        return (self._moment + 1) % self._total_moment


class PatrickStarManager(metaclass=SingletonMeta):
    """
    知道所有设备的使用情况，来指导payload的迁移
    singleton类，被所有进程访问
    拥有所有chunk信息的overview picture
    """
    def __init__(self):
        self.gpu_chunk_available_mem = 0
        self.cpu_chunk_available_mem = 0

        self.gpu_chunk_used_mem = 0
        self.cpu_chunk_used_mem = 0

        args = get_args()
        if args.use_fake_dist:
            rank = 0
            # 获得系统的存储信息
            self._overall_gpu_mem = torch.cuda.get_device_properties(
                rank).total_memory * 0.6 / torch.distributed.get_world_size()
            self._overall_cpu_mem = psutil.virtual_memory(
            ).total * 0.6 / torch.distributed.get_world_size()
        else:
            rank = args.local_rank
            # 获得系统的存储信息
            self._overall_gpu_mem = torch.cuda.get_device_properties(
                rank).total_memory * 0.6
            self._overall_cpu_mem = psutil.virtual_memory(
            ).total * 0.6 / torch.distributed.get_world_size()

        # 统计信息
        self.cpu_used_list = []
        self.cpu_chunk_used_list = []
        # non-chunk memory
        self.cpu_sys_used_list = []

        self.gpu_used_list = []
        self.gpu_chunk_used_list = []
        self.gpu_sys_used_list = []

        # 节拍器
        self.metronome = Metronome()

        # 预热标志
        self.warmup = False
        self._start_training = False

    def start_train(self, is_warmup):
        self.warmup = is_warmup
        self._start_training = True
        logger.info(f'Start to train. Manager sets warmup {is_warmup}')

    def reset_metronome(self):
        if self.warmup is True:
            self.warmup = False
            logger.info(
                f'***************** WARMUP PHASE OVER *****************')
        self.metronome.reset()
        logger.info('Manager Resets Metronome')

    def tiktac(self, client):
        """
        打节拍，同时记录此刻的内存使用情况
        """
        args = get_args()
        if torch.distributed.is_initialized():
            rank = args.local_rank
        else:
            rank = 0
        gpu_device = torch.device(f'cuda:{rank}')
        cpu_device = torch.device('cpu:0')

        if self.warmup:
            gpu_used = get_sys_memory_used(gpu_device)

            self.gpu_used_list.append(gpu_used)
            # 精确地统计chunk used memory
            self.gpu_chunk_used_list.append(self.gpu_chunk_used_mem)
            self.gpu_sys_used_list.append((gpu_used - self.gpu_chunk_used_mem))

            # 用来计算还可以分配多少内存
            # 希望获得系统内存+chunk used memory的总和
            cpu_used = get_sys_memory_used(cpu_device)
            self.cpu_used_list.append(cpu_used)
            self.cpu_chunk_used_list.append(self.cpu_chunk_used_mem)
            self.cpu_sys_used_list.append((cpu_used - self.cpu_chunk_used_mem))
        else:
            # 非warmup需要对Chunk Memory调仓
            # 如果下一刻的Chunk Memory可用空间小于当前Chunk Memory
            # 则需要从设备内存移出Chunk
            next_mom = self.metronome.next_moment()
            cur_mom = self.metronome.moment()
            gpu_next_mom_ava_chunk_mem = self._overall_gpu_mem - self.gpu_sys_used_list[
                next_mom]
            gpu_cur_mom_used_chunk_mem = client.chunk_list.get_chunk_memory_used(
                gpu_device)
            # logger.info(f'gpu_cur_mom_used_chunk_mem {gpu_cur_mom_used_chunk_mem/1e6} MB gpu_next_mom_ava_chunk_mem {gpu_next_mom_ava_chunk_mem/1e6} MB')
            if gpu_next_mom_ava_chunk_mem < gpu_cur_mom_used_chunk_mem:
                offload_size = gpu_cur_mom_used_chunk_mem - gpu_next_mom_ava_chunk_mem
                # logger.info(
                #     f'available memory before room making {(self._overall_gpu_mem - torch.cuda.memory_allocated())/1e6} MB on gpu'
                # )
                logger.info(
                    f'Making {offload_size/1e6} MB space on gpu, gpu_cur_mom_used_chunk_mem {gpu_cur_mom_used_chunk_mem/1e6} MB gpu_next_mom_ava_chunk_mem {gpu_next_mom_ava_chunk_mem/1e6} MB'
                )
                client.chunk_list.make_room(offload_size, gpu_device)

            # # 对CPU的Chunk Mem进行调仓
            # cpu_next_mom_ava_chunk_mem = self._overall_cpu_mem - self.cpu_sys_used_list[
            #     next_mom]
            # cpu_cur_mom_used_chunk_mem = client.chunk_list.get_chunk_memory_used(
            #     cpu_device)
            # # logger.info(
            # #     f'cpu_cur_mom_used_chunk_mem {cpu_cur_mom_used_chunk_mem/1e6} MB cpu_next_mom_ava_chunk_mem {cpu_next_mom_ava_chunk_mem/1e6} MB'
            # # )
            # # 当前时刻used chunk mem，下一时刻ava chunk mem = used chunk mem + free chunk mem
            # # 降低used chunk mem，使之可以达到下一时刻ava chunk mem
            # if cpu_next_mom_ava_chunk_mem < cpu_cur_mom_used_chunk_mem:
            #     offload_size = cpu_cur_mom_used_chunk_mem - cpu_next_mom_ava_chunk_mem
            #     # logger.info(f'chunk list has to make room {offload_size/1e6} MB')
            #     client.chunk_list.make_room(offload_size, cpu_device)

            # client.chunk_list.make_room(cpu_device)
        # logger.info(f'available memory {(self._overall_gpu_mem - torch.cuda.memory_allocated())/1e6} MB on gpu')
        self.metronome.tiktac()

    def add(self, device_type: str, size_in_bytes: int):
        """
        登记，设备device_type:index增加size个bytes内存使用
        """
        if device_type == "cpu":
            self.cpu_chunk_used_mem += size_in_bytes
        elif device_type == "cuda":
            # logger.info(f'use chunk memory {size_in_bytes} on gpu')
            self.gpu_chunk_used_mem += size_in_bytes
        else:
            raise f"device type {device_type} is not supported"

    def delete(self, device_type, size_in_bytes):
        """
        checkout，设备device_type:index减少size个bytes内存使用
        """
        if device_type == "cpu":
            self.cpu_chunk_used_mem -= size_in_bytes
        elif device_type == "cuda":
            self.gpu_chunk_used_mem -= size_in_bytes
        else:
            raise f"device type {device_type} is not supported"

    def free_chunk_mem(self, device_type):
        """
        可以用来分配的Chunk空闲内存，派出已经分配的内存
        """
        size = self.available_chunk_mem(device_type) - self.used_chunk_mem(
            device_type)
        # logger.info(
        #     f'free_chunk_mem on {device_type} {size/1e6} MB on mement {self.metronome.moment()}'
        # )
        return size

    def used_chunk_mem(self, device_type):
        if device_type == "cpu":
            return self.cpu_chunk_used_mem
        elif device_type == "cuda":
            return self.gpu_chunk_used_mem

    def available_chunk_mem(self, device_type):
        """
        返回用可以于分配Chunk的内存，即可用内存。
        可用内存包括已经分配被Chunk占据的内存和闲置(free)的内存。
        预热阶段是三分之一GPU内存和全部CPU内存。
        非预热阶段，是当前moment和下一moment可用内存的最小值。
        """
        if device_type == "cpu":
            if self.warmup or not self._start_training:
                # TODO(jiaruifang)瞎拍一个数，预热阶段三分之一GPU显存用来存储chunk
                return self._overall_cpu_mem
            else:
                return self._overall_cpu_mem
                # next_mem = self.metronome.next_moment()
                # next_mom_ava_mem = self._overall_cpu_mem - self.cpu_sys_used_list[
                #     next_mem]
                # cur_mom_ava_mem = self._overall_cpu_mem - self.cpu_sys_used_list[
                #     self.metronome.moment()]
                # # TODO(jiaruifang）
                # ava_cpu_chunk_mem = min(next_mom_ava_mem, cur_mom_ava_mem)
                # logger.info(
                #     f'available cpu chunk memory is {ava_cpu_chunk_mem/1e6} MB'
                # )
                # return ava_cpu_chunk_mem
        elif device_type == "cuda":
            if self.warmup or not self._start_training:
                # TODO(jiaruifang)瞎拍一个数，预热阶段三分之一GPU显存用来存储chunk
                return self._overall_gpu_mem / 3
            else:
                next_mom = self.metronome.next_moment()
                cur_mom = self.metronome.moment()
                next_mom_ava_mem = self._overall_gpu_mem - self.gpu_sys_used_list[
                    next_mom]
                cur_mom_ava_mem = self._overall_gpu_mem - self.gpu_sys_used_list[
                    cur_mom]
                return min(next_mom_ava_mem, cur_mom_ava_mem)

    def show_mem_curve(self):
        with open('gpu_used_curve.txt', 'w') as fh:
            fh.write(
                f'gpu_chunk_used_list {len(self.gpu_chunk_used_list)} \n {list(map(lambda x : x/1e6, self.gpu_chunk_used_list))}\n'
            )
            fh.write(
                f'gpu_sys_used_list {list(map(lambda x: x/1e6, self.gpu_sys_used_list))}\n'
            )
            fh.write(
                f'gpu_used_list \n {list(map(lambda  x: x/1e6, self.gpu_used_list))}\n'
            )

        with open('cpu_used_curve.txt', 'w') as fh:
            fh.write(
                f'cpu_chunk_used_list {len(self.gpu_chunk_used_list)} \n {list(map(lambda x : x/1e6, self.cpu_chunk_used_list))}\n'
            )
            fh.write(
                f'cpu_sys_used_list {list(map(lambda x: x/1e6, self.cpu_sys_used_list))}\n'
            )
            fh.write(
                f'cpu_used_list \n {list(map(lambda  x: x/1e6, self.cpu_used_list))}\n'
            )
