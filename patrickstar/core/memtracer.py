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

import time

import torch

from patrickstar.core.const import TrainingStage
from patrickstar.utils import (
    log_dist,
    get_sys_memory_info,
    get_sys_memory_used,
    get_local_world_size,
)
from concurrent.futures import ThreadPoolExecutor


class AsyncMemoryMonitor:
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
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                get_sys_memory_used(
                    torch.device(f"cuda:{torch.cuda.current_device()}")
                ),
            )
            time.sleep(self.interval)

        return max_usage


class MemoryInfo:
    def __init__(self, cpu_total, cpu_chunk, cpu_sys, gpu_total, gpu_chunk, gpu_sys):
        self.cpu_total = cpu_total
        self.cpu_chunk = cpu_chunk
        self.cpu_sys = cpu_sys

        self.gpu_total = gpu_total
        self.gpu_chunk = gpu_chunk
        self.gpu_sys = gpu_sys


class RuntimeMemTracer:
    r"""Collecting memory statistics on CPU and GPU during training,
    to direct chunk moving.
    Glossary:
        Chunkable Memry: Memory can be used to store chunk.
    """

    def __init__(self, local_rank, metronome, config=None):
        self.local_rank = local_rank
        self.metronome = metronome

        self.chunk_used_mem = {"cpu": 0, "cuda": 0}
        self.cpu_chunk_used_mem_pinned = 0

        assert config is not None
        self.overall_gpu_mem_ratio = config.get("overall_gpu_mem_ratio", 0.8)
        self.overall_cpu_mem_ratio = config.get("overall_cpu_mem_ratio", 0.8)
        self._margin_use_ratio = config.get("margin_use_ratio", 0.8)
        self.warmup_gpu_chunk_mem_ratio = config.get("warmup_gpu_chunk_mem_ratio", 0.1)
        self.use_async_mem_monitor = config.get("use_async_mem_monitor", False)
        if self.use_async_mem_monitor:
            self.async_mem_monitor = AsyncMemoryMonitor()

        mem_info = get_sys_memory_info()
        local_world_size = get_local_world_size()
        self.overall_gpu_mem = (
            torch.cuda.get_device_properties(self.local_rank).total_memory
            * self.overall_gpu_mem_ratio
        )
        self.overall_cpu_mem = (
            mem_info.total * self.overall_cpu_mem_ratio / local_world_size
        )

        log_dist(
            f"Init Manager over all gpu mem {self.overall_gpu_mem / 1e6} MB, "
            f"cpu mem {self.overall_cpu_mem / 1e6} MB"
        )
        self.memory_stats = []

        self.chunk_size = 0
        self.max_cpu_sys_used = 0

    def start(self, chunk_size):
        self.chunk_size = chunk_size
        if self.use_async_mem_monitor:
            self.async_mem_monitor.start()
        log_dist("**** Memory Tracer started ****")

    def end(self):
        if self.use_async_mem_monitor:
            self.async_mem_monitor.finish()
        log_dist("**** Memory Tracer ended ****")

    def trace(self):
        """Record the memory usage of the moment and increase moment counter."""
        if self.metronome.is_warmup:
            gpu_device = torch.device(f"cuda:{self.local_rank}")
            cpu_device = torch.device("cpu:0")
            gpu_used = get_sys_memory_used(gpu_device)
            # Get peak memory between cur tracing and the prev tracing
            if self.use_async_mem_monitor:
                max_mem_period = self.async_mem_monitor.finish()
                gpu_used = max(max_mem_period, gpu_used)
                self.async_mem_monitor.start()

            cpu_used = get_sys_memory_used(cpu_device)
            self.memory_stats.append(
                MemoryInfo(
                    cpu_total=cpu_used,
                    cpu_chunk=self.chunk_used_mem["cpu"],
                    # detected cpu memory usage (already excluded pinned memory) - chunk non
                    # pinned memory usage = system cpu usage (non-chunk cpu memory)
                    cpu_sys=cpu_used
                    - (self.chunk_used_mem["cpu"] - self.cpu_chunk_used_mem_pinned),
                    gpu_total=gpu_used,
                    gpu_chunk=self.chunk_used_mem["cuda"],
                    gpu_sys=gpu_used - self.chunk_used_mem["cuda"],
                )
            )
            cur_mom = self.metronome.moment
            assert len(self.memory_stats) - 1 == cur_mom

        self.metronome.tiktac()

    def add(self, device_type, size_in_bytes, is_pinned=False):
        self.chunk_used_mem[device_type] += size_in_bytes
        if device_type == "cpu" and is_pinned:
            self.cpu_chunk_used_mem_pinned += size_in_bytes

    def delete(self, device_type, size_in_bytes, is_pinned=False):
        self.chunk_used_mem[device_type] -= size_in_bytes
        if device_type == "cpu" and is_pinned:
            self.cpu_chunk_used_mem_pinned -= size_in_bytes

    def reset_memory_stats(self):
        """
        Reset statistics collected from memory tracing.
        It is used in case of gradient overflow during warmup and
        the memory stats is incomplete.
        """
        # As the reset happens right before forward, if the manager
        # is still doing warmup, it means the previous run didn't
        # cover the full procedure (forward -> backward -> optimizer).
        # Therefore clean the stats collected.
        if self.metronome.is_warmup:
            self.memory_stats = []
        log_dist("Reset Memory Statistics")

    def remaining_chunk_mem(self, device_type):
        """
        Return the remainig chunkable memory on device_type,
        which can be used to host chunks.
        """
        available_mem = self.available_chunk_mem(device_type)
        chunk_mem = self.chunk_used_mem[device_type]
        if device_type == "cuda":
            print(
                "available_mem:",
                available_mem / 1024 ** 2,
                "chunk_mem:",
                chunk_mem / 1024 ** 2,
            )
        return available_mem - chunk_mem

    def available_chunk_mem(self, device_type):
        r"""The amount of memory on device_type that can be used for chunks.
        A.k.a chunkale memory.
        This includes the used memory that has been allocated for chunks
        and the remaining memory.

            available_chunk_mem = remaining_chunk_mem + used_chunk_mem

        In warmup, the available chunk mem is part of GPU mem and all
        CPU mem.
        After warmup, it is the minimal value of available mem of the
        current moment and next moment.
        """
        # If the training is not started, ava chunk mem is the overall system mem.
        if self.metronome.training_stage == TrainingStage.UNSTART:
            if device_type == "cpu":
                return self.overall_cpu_mem
            elif device_type == "cuda":
                return self.overall_gpu_mem

        # If it is warmup stage, chunk can used gpu_ratio * overall_gpu
        # chunk can used all cpu.
        if self.metronome.is_warmup:
            if device_type == "cpu":
                return self.overall_cpu_mem
            elif device_type == "cuda":
                return self.overall_gpu_mem * self.warmup_gpu_chunk_mem_ratio

        if device_type == "cpu":
            local_world_size = get_local_world_size()
            if self.metronome.training_stage != TrainingStage.ADAM:
                return self.overall_cpu_mem - self.max_cpu_sys_used / local_world_size
            else:
                return self.overall_cpu_mem
        elif device_type == "cuda":
            next_mom = self.metronome.next_moment()
            cur_mom = self.metronome.moment
            ava_mem = max(
                self.memory_stats[next_mom].gpu_sys, self.memory_stats[cur_mom].gpu_sys
            )
            return (self.overall_gpu_mem - ava_mem) * 0.5
