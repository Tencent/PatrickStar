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
import logging


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


class HybridPSManager(metaclass=SingletonMeta):
    """
  知道所有设备的使用情况，来指导payload的迁移
  singleton类，被所有进程访问
  拥有所有chunk信息的overview picture
  """
    def __init__(self):
        mp_manager = Manager()
        self._is_init_ = mp_manager.Value('_is_init_', False)
        self.gpu_max_mem_list = mp_manager.list([])
        self.cpu_max_mem_list = mp_manager.list([])
        self.gpu_used_mem_list = mp_manager.list([])
        self.cpu_used_mem_list = mp_manager.list([])

        self.cpu_mem_usage_curve = []
        self.gpu_mem_usage_curve = []

    def init(self, gpu_info, cpu_info):
        if self._is_init_.value:
            self.reset(gpu_info, cpu_info)
            return

        for item in gpu_info:
            self.gpu_max_mem_list.append(item)
            self.gpu_used_mem_list.append(0)

        for item in cpu_info:
            self.cpu_max_mem_list.append(item)
            self.cpu_used_mem_list.append(0)
        self._is_init_.value = True

    def reset(self, gpu_info, cpu_info):
        mp_manager = Manager()
        self._is_init_ = mp_manager.Value('_is_init_', False)
        self.gpu_max_mem_list = mp_manager.list([])
        self.cpu_max_mem_list = mp_manager.list([])
        self.gpu_used_mem_list = mp_manager.list([])
        self.cpu_used_mem_list = mp_manager.list([])
        self.init(gpu_info, cpu_info)

    def is_init(self):
        return self._is_init_.value

    def visit(self):
        for idx, (used_mem, max_mem) in enumerate(
                zip(self.gpu_used_mem_list, self.gpu_max_mem_list)):
            print(f"GPU:{idx} used mem {used_mem} B max mem {max_mem} B")
        for idx, (used_mem, max_mem) in enumerate(
                zip(self.cpu_used_mem_list, self.cpu_max_mem_list)):
            print(f"CPU:{idx} used mem {used_mem} B max mem {max_mem} B")

    def add(self, device_type: str, index: int, size: int):
        """
        登记，设备device_type:index增加size个bytes内存使用
        """
        if index is None:
            index = 0

        if device_type == "cpu":
            self.cpu_used_mem_list[index] += size
            self.cpu_mem_usage_curve.append(self.used_mem(device_type, index))
        elif device_type == "cuda":
            self.gpu_used_mem_list[index] += size
            self.gpu_mem_usage_curve.append(self.used_mem(device_type, index))
        else:
            raise f"device type {device_type} is not supported"

    def delete(self, device_type, index, size):
        """
        checkout，设备device_type:index减少size个bytes内存使用
        """
        if index is None:
            index = 0

        if device_type == "cpu":
            self.cpu_used_mem_list[index] -= size
            self.cpu_mem_usage_curve.append(
                self.available_mem(device_type, index))
        elif device_type == "cuda":
            self.gpu_used_mem_list[index] -= size
            self.gpu_mem_usage_curve.append(self.used_mem(device_type, index))
        else:
            raise f"device type {device_type} is not supported"

    def schedule(self, size_in_byte: int, refer_dev_idx: int):
        """
        找到一个设备，可以分配size_in_byte个bytes存储空间
        refer_dev_idx, 调用进程管理的gpu编号
        """
        if self.available_mem("cpu", 0) >= size_in_byte:
            return torch.device("cpu")
        elif self.available_mem("cuda", refer_dev_idx) >= size_in_byte:
            return torch.device(f"cuda:{refer_dev_idx}")
        else:
            for idx in range(self.gpu_num()):
                if idx == refer_dev_idx:
                    pass
                if self.available_mem("cuda", idx) >= size_in_byte:
                    # self.add("cuda", idx, size_in_byte)
                    return torch.device(f"cuda:{idx}")
        logging.error(f"HybridPSManager can not find {size_in_byte} space")
        raise RuntimeError

    def available_mem(self, device_type, index):
        index = 0 if index is None else index
        if device_type == "cuda":
            return self.gpu_max_mem_list[index] - self.gpu_used_mem_list[index]
        elif device_type == "cpu":
            return self.cpu_max_mem_list[index] - self.cpu_used_mem_list[index]

    def gpu_num(self):
        return len(self.gpu_max_mem_list)

    def cpu_num(self):
        return len(self.cpu_max_mem_list)

    def used_mem(self, device_type, index):
        if device_type == "cpu":
            return self.cpu_used_mem_list[index]
        elif device_type == "cuda":
            return self.gpu_used_mem_list[index]

    def max_mem(self, device_type, index):
        index = 0 if index is None else index
        if device_type == "cpu":
            return self.cpu_max_mem_list[index]
        elif device_type == "cuda":
            return self.gpu_max_mem_list[index]


if __name__ == "__main__":
    s1 = HybridPSManager()
    s1.init([64, 64], [128])

    # do nothing if you initialize a singleton twice
    s2 = HybridPSManager()
    s2.init([32, 32, 3], [32])
    assert s2.gpu_num() == 2

    if id(s1) == id(s2):
        print(
            "HybridPSManager works, both variables contain the same instance.")
    else:
        print("HybridPSManager failed, variables contain different instances.")
