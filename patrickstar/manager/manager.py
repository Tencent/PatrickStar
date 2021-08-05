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
import psutil
import logging as logger
from torch.multiprocessing import Process, Manager

from patrickstar.utils.memory_monitor import get_sys_memory_used
from patrickstar.deepspeed_helper.global_vars import get_args
from patrickstar.core.const import TrainingStage


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

    def get_total_mom(self):
        assert self._total_moment is not None, f"Don not use get_total during warmup"
        return self._total_moment

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

        # 需要设置的超参数
        self._overall_gpu_mem_ratio = args.overall_gpu_mem_ratio
        self._overall_cpu_mem_ratio = args.overall_cpu_mem_ratio
        self._margin_use_ratio = args.margin_use_ratio 
        self.warmup_gpu_chunk_mem_ratio = args.warmup_gpu_chunk_mem_ratio

        if args.use_fake_dist:
            rank = 0
            # 伪分布式训练是，大家共享一块GPU
            self._overall_gpu_mem = torch.cuda.get_device_properties(
                rank
            ).total_memory * self._overall_gpu_mem_ratio / torch.distributed.get_world_size(
            )
            self._overall_cpu_mem = psutil.virtual_memory(
            ).total * self._overall_cpu_mem_ratio / torch.distributed.get_world_size(
            )
        else:
            rank = args.local_rank
            # 获得系统的存储信息
            self._overall_gpu_mem = torch.cuda.get_device_properties(
                rank).total_memory * self._overall_gpu_mem_ratio
            self._overall_cpu_mem = psutil.virtual_memory(
            ).total * self._overall_cpu_mem_ratio / torch.distributed.get_world_size(
            )

        logger.info(
            f'Init Manager over all gpu mem {self._overall_gpu_mem/1e6} MB, cpu mem {self._overall_cpu_mem/1e6} MB'
        )
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
        self._training_stage = TrainingStage.UNSTART
        # 总gpu可用内存 减去 峰值系统内存占用 减去 paramfp16峰值
        self._margin_chunk_num_for_gpu_adam = 0
        self._default_chunk_size = 0

    def is_warmup_training(self):
        return self._start_training and self.warmup

    def is_nonwarmup_training(self):
        return self._start_training and not self.warmup

    def start_train(self, is_warmup, param_fp16_chunk_size, chunk_size):
        self.warmup = is_warmup
        self._start_training = True
        self._training_stage = TrainingStage.FWD
        self._param_fp16_chunk_size = param_fp16_chunk_size
        self._default_chunk_size = chunk_size
        logger.info(f'Start to train. Manager sets warmup {is_warmup}')

    def update_margin_mem(self):
        """
        更新GPU内剩余的空间可存储的Chunk数目
        """
        max_gpu_sys_used = max(self.gpu_sys_used_list)
        margin_mem_size = self._overall_gpu_mem - max_gpu_sys_used - self._param_fp16_chunk_size
        # 12 = 4 + 4 + 4 fp32 + m + v
        self._margin_chunk_num_for_gpu_adam = (margin_mem_size) / (
            self._default_chunk_size * 12) * self._margin_use_ratio

        logger.info("*********** GPU INFO AFTER BWD ***************")
        logger.info(
            f'Max GPU System Mem (non-chunk) Used {max(self.gpu_sys_used_list)/1e6} MB'
        )
        logger.info(
            f'Param FP16 Chunk Size {self._param_fp16_chunk_size/1e6} MB')
        logger.info(
            f'Margin Mem Size {margin_mem_size/1e6} MB, available chunk num for Optimizer States {self._margin_chunk_num_for_gpu_adam}'
        )
        logger.info(f'OVERALL GPU MEM {self._overall_gpu_mem}')

    def reset_metronome(self):
        """
        重置节拍器
        """
        if self.warmup is True:
            self.warmup = False
            logger.info(
                f'***************** WARMUP PHASE OVER *****************')

        self.metronome.reset()
        logger.info('Manager Resets Metronome')

    def get_margin_chunk_num_for_gpu_adam(self):
        return self._margin_chunk_num_for_gpu_adam

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
        gpu_used = get_sys_memory_used(gpu_device)

        if self.warmup:
            self.gpu_used_list.append(gpu_used)
            # 精确地统计chunk used memory
            self.gpu_chunk_used_list.append(self.gpu_chunk_used_mem)
            self.gpu_sys_used_list.append((gpu_used - self.gpu_chunk_used_mem))

            cpu_used = get_sys_memory_used(cpu_device)
            self.cpu_used_list.append(cpu_used)
            self.cpu_chunk_used_list.append(self.cpu_chunk_used_mem)
            self.cpu_sys_used_list.append((cpu_used - self.cpu_chunk_used_mem))

            # 非warmup迭代时按照cur_mom下标更新，warmup迭代更新在list末尾。
            # 确保list最后一个元素和cur_mom和此时更新的下标一致
            cur_mom = self.metronome.moment()
            assert len(self.gpu_sys_used_list) - 1 == cur_mom
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
            if gpu_next_mom_ava_chunk_mem < gpu_cur_mom_used_chunk_mem:
                offload_size = gpu_cur_mom_used_chunk_mem - gpu_next_mom_ava_chunk_mem
                # Note 触发GPU-CPU内存移动
                client.chunk_list.make_room(offload_size, gpu_device)

            # 每个节拍都校准gpu sys used
            self.gpu_sys_used_list[
                cur_mom] = gpu_used - self.gpu_chunk_used_mem

        self.metronome.tiktac()

    def get_cur_mom(self):
        return self.metronome.moment()

    def get_total_mom(self):
        return self.metronome.get_total_mom()

    def add(self, device_type: str, size_in_bytes: int):
        """
        登记，设备device_type:index增加size个bytes内存使用
        """
        if device_type == "cpu":
            self.cpu_chunk_used_mem += size_in_bytes
        elif device_type == "cuda":
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
        logger.debug(
            f'free_chunk_mem on {device_type} {size/1e6} MB on mement {self.metronome.moment()}'
        )
        return size

    def used_chunk_mem(self, device_type):
        if device_type == "cpu":
            return self.cpu_chunk_used_mem
        elif device_type == "cuda":
            return self.gpu_chunk_used_mem
        else:
            raise RuntimeError(f"used_chunk_mem {device_type}")

    def available_chunk_mem(self, device_type):
        """
        返回用可以于分配Chunk的内存，即可用内存。
        可用内存包括已经分配被Chunk占据的内存和闲置(free)的内存。
        available_chunk_mem = free_chunk_mem + used_chunk_mem
        预热阶段是部分GPU内存和全部CPU内存。
        非预热阶段，是当前moment和下一moment可用内存的最小值。
        """
        args = get_args()
        if device_type == "cpu":
            if self.warmup or not self._start_training:
                # TODO(jiaruifang)瞎拍一个数，预热阶段三分之一GPU显存用来存储chunk
                return self._overall_cpu_mem
            else:
                return self._overall_cpu_mem
        elif device_type == "cuda":
            if args.always_warmup or self.warmup or not self._start_training:
                if self._training_stage == TrainingStage.ADAM:
                    # ADAM时没有activation所以显存可以全部给Chunk，需要两个default chunk size做buffer，这里先预留6个
                    ava_mem = self._overall_gpu_mem - 4 * self._default_chunk_size * 4
                    logger.debug(
                        f'GPU available_chunk_mem is {ava_mem/1e6} MB')
                    return ava_mem
                else:
                    # TODO(jiaruifang)瞎拍一个数，预热阶段三分之一GPU显存用来存储chunk
                    return self._overall_gpu_mem * self.warmup_gpu_chunk_mem_ratio
            else:
                if self._training_stage == TrainingStage.ADAM:
                    return self._overall_gpu_mem - 4 * self._default_chunk_size * 4
                elif self._training_stage == TrainingStage.FWD:
                    next_mom = self.metronome.next_moment()
                    cur_mom = self.metronome.moment()
                    next_mom_ava_mem = self._overall_gpu_mem - 1.5 * self.gpu_sys_used_list[
                        next_mom]
                    cur_mom_ava_mem = self._overall_gpu_mem - 1.5 * self.gpu_sys_used_list[
                        cur_mom]
                    return min(next_mom_ava_mem, cur_mom_ava_mem) - args.world_size * 2 * self._default_chunk_size
                elif self._training_stage == TrainingStage.BWD:
                    next_mom = self.metronome.next_moment()
                    cur_mom = self.metronome.moment()
                    next_mom_ava_mem = self._overall_gpu_mem - 2 * self.gpu_sys_used_list[
                        next_mom]
                    cur_mom_ava_mem = self._overall_gpu_mem - 2 * self.gpu_sys_used_list[
                        cur_mom]
                    return min(next_mom_ava_mem, cur_mom_ava_mem) - args.world_size * 2 * self._default_chunk_size

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
            # fh.write(
            #     f'cpu_sys_used_list {list(map(lambda x: x/1e6, self.cpu_sys_used_list))}\n'
            # )
            # fh.write(
            #     f'cpu_used_list \n {list(map(lambda  x: x/1e6, self.cpu_used_list))}\n'
            # )
