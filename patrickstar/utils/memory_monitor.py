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

import gc

import psutil
import torch
from .distributed import get_rank, get_local_world_size
from .memory import get_sys_memory_info


def get_sys_memory_used(device):
    """
    Get the free memory info of device.
    Notice that for CPU, this function will return 1/N of the total free memory,
    where N is the world size.
    """
    if device.type == "cuda":
        ret = torch.cuda.memory_allocated()
        # get the peak memory to report correct data, so reset the counter for the next call
        if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
            torch.cuda.reset_peak_memory_stats()
    elif device.type == "cpu":
        mem_info = get_sys_memory_info()
        ret = mem_info.used / get_local_world_size()
    return ret


def see_memory_usage(message):
    if not get_rank() == 0:
        return

    gc.collect()

    scale = 1024 ** 2
    # Print message except when distributed but not rank 0
    print(message)
    print(
        f"memory allocated {round(torch.cuda.memory_allocated() / scale, 2)} MB "
        f"max memory allocated {round(torch.cuda.max_memory_allocated() / scale, 2)} MB "
        f"memory reserved {round(torch.cuda.memory_reserved() / scale, 2)} MB "
        f"max memory reserved {round(torch.cuda.max_memory_reserved() / scale)} MB "
    )

    # TODO(zilinzhu) Find how to get the available and percent value of the
    # memory in docker to substitute psutil.virtual_memory to get_sys_memory_info.
    vm_stats = psutil.virtual_memory()
    used_gb = round((vm_stats.total - vm_stats.available) / 1024 ** 3, 2)
    print(f"CPU Virtual Memory: used = {used_gb} GB, percent = {vm_stats.percent}%")

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()
