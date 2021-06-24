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
import logging as logger


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


class PatrickStarManager(metaclass=SingletonMeta):
    """
  知道所有设备的使用情况，来指导payload的迁移
  singleton类，被所有进程访问
  拥有所有chunk信息的overview picture
  """
    def __init__(self):
        self.gpu_max_mem = 0
        self.cpu_max_mem = 0
        self.gpu_used_mem = 0
        self.cpu_used_mem = 0
        self.cpu_mem_usage_curve = []
        self.gpu_mem_usage_curve = []
        self._is_init_ = False

    def is_init(self):
        return self._is_init_

    def init(self, max_gpu_memory, max_cpu_memory):
        self.gpu_max_mem = max_gpu_memory
        self.cpu_max_mem = max_cpu_memory
        logger.info(
            'Init Manager with gpu max mem {self.gpu_max_mem} and cpu max mem {self.cpu_max_mem}'
        )
        self._is_init_ = True

    def reset(self, max_gpu_memory, max_cpu_memory):
        self.init(max_gpu_memory, max_cpu_memory)

    def visit(self):
        logger.info(
            f"CPU used mem {self.cpu_used_mem} B, GPU used mem {self.gpu_used_mem} B"
        )

    def add(self, device_type: str, size_in_bytes: int):
        """
        登记，设备device_type:index增加size个bytes内存使用
        """
        if device_type == "cpu":
            self.cpu_used_mem += size_in_bytes
            self.cpu_mem_usage_curve.append(self.cpu_used_mem)
        elif device_type == "cuda":
            self.gpu_used_mem += size_in_bytes
            self.gpu_mem_usage_curve.append(self.gpu_used_mem)
        else:
            raise f"device type {device_type} is not supported"

    def delete(self, device_type, size_in_bytes):
        """
        checkout，设备device_type:index减少size个bytes内存使用
        """
        if device_type == "cpu":
            self.cpu_used_mem -= size_in_bytes
            self.cpu_mem_usage_curve.append(self.cpu_used_mem)
        elif device_type == "cuda":
            self.gpu_used_mem -= size_in_bytes
            self.gpu_mem_usage_curve.append(self.gpu_used_mem)
        else:
            raise f"device type {device_type} is not supported"

    def available_mem(self, device_type):
        if device_type == "cuda":
            return self.gpu_max_mem - self.gpu_used_mem
        elif device_type == "cpu":
            return self.cpu_max_mem - self.cpu_used_mem

    def used_mem(self, device_type):
        if device_type == "cpu":
            return self.cpu_used_mem
        elif device_type == "cuda":
            return self.gpu_used_mem

    def max_mem(self, device_type):
        if device_type == "cpu":
            return self.cpu_max_mem
        elif device_type == "cuda":
            return self.gpu_max_mem
