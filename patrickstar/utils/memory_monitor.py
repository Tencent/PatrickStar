# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gc

import psutil
import torch
from time import sleep
from .distributed import get_rank, get_world_size
from .memory import get_memory_info
from .singleton_meta import SingletonMeta
from concurrent.futures import ThreadPoolExecutor


class AsyncMemoryMonitor(metaclass=SingletonMeta):
    def __init__(self, power=3):
        """
        An Async Mem Monitor runing during computing.
        Sampling GPU memory usage of the current GPU dev
        at interval of 1/(10**power) sec.
        """
        self.keep_measuring = False
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.monitor_thread = None
        self.interval = 1 / (10 ** power)

    def set_interval(self, power: int):
        self.interval = 1 / (10 ** power)

    def start(self):
        self.keep_measuring = True
        self.monitor_thread = self.executor.submit(self._measure_usage)

    def finish(self):
        if self.keep_measuring is False:
            return 0
        self.keep_measuring = False
        max_usage = self.monitor_thread.result()
        self.monitor_thread = None
        return max_usage

    def _measure_usage(self):
        max_usage = 0
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                # resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                get_sys_memory_used(dev),
            )
            # print(f"tmp max usage {max_usage}")
            sleep(self.interval)

        return max_usage


def close_asyn_mem_monitor():
    monitor = AsyncMemoryMonitor()
    monitor.finish()


def max_mem_usage_period(interval: int = None):
    """
    A function used to find the max memory usage during a period time by sampling,
    between now and the perivous call of this function.
    ret:
        the max GPU memory used during this period
    """
    monitor = AsyncMemoryMonitor()
    max_mem = monitor.finish()
    monitor.start()
    return max_mem


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
        mem_info = get_memory_info()
        ret = mem_info.used / get_world_size()
    return ret


def see_memory_usage(message, force=False, scale_name="MB"):
    if not force:
        return
    if not get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    if scale_name == "MB":
        scale = 1024 * 1024
    elif scale_name == "B":
        scale = 1
    # Print message except when distributed but not rank 0
    print(message)
    print(
        f"MA {round(torch.cuda.memory_allocated() / scale, 2)} {scale_name} \
        Max_MA {round(torch.cuda.max_memory_allocated() / scale, 2)} {scale_name} \
        CA {round(torch.cuda.memory_reserved() / scale, 2)} {scale_name} \
        Max_CA {round(torch.cuda.max_memory_reserved() / scale)} {scale_name} "
    )

    # TODO(zilinzhu) Find how to get the available and percent value of the
    # memory in docker to substitute psutil.virtual_memory to get_memory_info.
    vm_stats = psutil.virtual_memory()
    used_gb = round(((vm_stats.total - vm_stats.available) / (1024 ** 3)), 2)
    print(f"CPU Virtual Memory: used = {used_gb} GB, percent = {vm_stats.percent}%")

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()
