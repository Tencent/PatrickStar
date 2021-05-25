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

# 统计chunk的lifecycle开关
import logging
from .memory_monitor import get_memory_used
import torch
from manager import HybridPSManager


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


class IterationTimer(metaclass=SingletonMeta):
    """
    记录每个access和release的时间点
    """
    def __init__(self):
        self.warmup = False
        self._moment = 0

        self.cpu_used_list = []
        self.cpu_ps_used_list = []
        self.gpu_used_list = []
        self.gpu_ps_used_list = []
        self.gpu_sys_used_list = []
        self.moment_device = {}

    def device_type(mom):
        return self.moment_device[mom]

    def tik(self, device_type=None):
        """
        时刻前进，只有access - release形成一个闭环是一个moment
        moment有一个计算设备，FWD BWD的设备是cuda
        adam的设备是cpu cuda都可以
        """
        self._moment += 1
        if self.warmup:
            if device_type == 'cuda':
                self.moment_device[self._moment] = 'cuda'
            else:
                self.moment_device[self._moment] = 'all'

        if self.warmup:
            mgr = HybridPSManager()
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
            gpu_device = torch.device(f'cuda:{rank}')
            gpu_ps_used = mgr.used_mem(gpu_device.type, gpu_device.index)
            gpu_used = get_memory_used(gpu_device)

            cpu_device = torch.device('cpu:0')
            cpu_ps_used = mgr.used_mem(gpu_device.type, gpu_device.index)
            cpu_used = get_memory_used(cpu_device)

            self.cpu_used_list.append(cpu_used / 1e6)
            self.cpu_ps_used_list.append(cpu_ps_used / 1e6)
            self.gpu_used_list.append(gpu_used / 1e6)
            self.gpu_ps_used_list.append(gpu_ps_used / 1e6)
            self.gpu_sys_used_list.append((gpu_used - gpu_ps_used) / 1e6)

    def moment(self):
        return self._moment

    def reset(self):
        self._moment = 0


# param访问
client_access_elapse = 0.
client_prepare_device_elapse = 0.
access_chunk_elapse = 0.
chunk_move_elapse = 0.
chunk_to_move_out_for_room_making_elapse = 0.
memory_allocate_elapse = 0.

# adam计算
cpu_adam_elapse = 0.
cpu_adam_f_elapse = 0.
cpu_adam_elapse = 0.

# param释放
client_release_elapse = 0.
memory_delete_elapse = 0.

# 数据移动
cpu_gpu_move_elapse = 0.
cpu_gpu_move_times = 0
cpu_gpu_move_data_amount = 0

gpu_cpu_move_elapse = 0.
gpu_cpu_move_times = 0
gpu_cpu_move_data_amount = 0

get_status_elapse = 0
cpu_adam_release_elapse = 0
cpu_adam_access_elapse = 0


def time_profiler():
    global client_access_elapse
    global client_prepare_device_elapse
    global access_chunk_elapse
    global chunk_move_elapse
    global chunk_to_move_out_for_room_making_elapse
    global memory_allocate_elapse

    global cpu_adam_elapse
    global cpu_adam_f_elapse
    global cpu_adam_elapse

    global client_release_elapse
    global memory_delete_elapse

    # 数据移动
    global cpu_gpu_move_elapse
    global cpu_gpu_move_times
    global cpu_gpu_move_data_amount

    global gpu_cpu_move_elapse
    global gpu_cpu_move_times
    global gpu_cpu_move_data_amount

    global get_status_elapse
    global cpu_adam_release_elapse
    global cpu_adam_access_elapse

    logging.info(f'CLIENT ACCESS ELAPSE')
    logging.info(f'* client_access_elapse {client_access_elapse} ')
    logging.info(f'** access_chunk_elapse {access_chunk_elapse}')
    logging.info(f'*** memory_allocate_elapse {memory_allocate_elapse}')
    logging.info(
        f'*** client_prepare_device_elapse {client_prepare_device_elapse}')
    logging.info(
        f'**** chunk_to_move_out_for_room_making_elapse {chunk_to_move_out_for_room_making_elapse}'
    )
    logging.info(f'**** chunk_move_elapse {chunk_move_elapse}')

    logging.info("DATA MOVE STATISTICS")
    logging.info(
        f'*** cpu_gpu_move_elapse {cpu_gpu_move_elapse} sec, times {cpu_gpu_move_times}, amount {cpu_gpu_move_data_amount/1e6} MB, Bandwidth {cpu_gpu_move_data_amount/1e6/(cpu_gpu_move_elapse + 1e-10)} MB/s'
    )
    logging.info(
        f'*** gpu_cpu_move_elapse {gpu_cpu_move_elapse} sec, times {gpu_cpu_move_times}, amount {gpu_cpu_move_data_amount/1e6} MB, Bandwidth {gpu_cpu_move_data_amount/1e6/(gpu_cpu_move_elapse + 1e-10)} MB/s'
    )

    logging.info("ADAM STATISTICS")
    logging.info(
        f'* cpu_adam_elapse {cpu_adam_elapse} cpu_adam_f_elapse {cpu_adam_f_elapse}'
    )
    logging.info(
        f'* cpu_adam_release_elapse {cpu_adam_release_elapse} cpu_adam_access_elapse {cpu_adam_access_elapse}'
    )

    logging.info(f'CLIENT RELASE ELAPSE')
    logging.info(f'* client_release_elapse {client_release_elapse}')
    logging.info(f'** memory_delete_elapse {memory_delete_elapse}')
    logging.info(f'*** get_status_elapse {get_status_elapse}')