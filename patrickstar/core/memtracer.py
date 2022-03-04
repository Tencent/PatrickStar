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

import time

import torch

from patrickstar.core.const import TrainingStage
from patrickstar.utils import (
    log_dist,
    get_sys_memory_info,
    get_sys_memory_used,
    get_local_world_size,
    logger,
    get_world_size,
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
        dev = torch.device(f"cuda:{torch.cuda.current_device()}")
        while self.keep_measuring:
            max_usage = max(
                max_usage,
                get_sys_memory_used(dev),
            )
            time.sleep(self.interval)

        return max_usage


class MemoryInfo:
    def __init__(self, cpu_total, cpu_chunk, cpu_sys, gpu_total, gpu_chunk, gpu_sys):
        self.cpu_total = cpu_total
        self.cpu_chunk = cpu_chunk
        # non-chunk memory
        self.cpu_sys = cpu_sys

        self.gpu_total = gpu_total
        self.gpu_chunk = gpu_chunk
        # non-chunk memory
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

        self.cpu_chunk_available_mem = 0
        self.cpu_chunk_used_mem = 0
        self.cpu_chunk_used_mem_pinned = 0

        self.gpu_chunk_available_mem = 0
        self.gpu_chunk_used_mem = 0

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
        self.memory_infos = []

        self.chunk_size = 0
        self.max_cpu_sys_used = 0

    def close_tracer(self):
        """
        Close memory tracer.
        """
        if self.use_async_mem_monitor:
            self.async_mem_monitor.finish()
        log_dist("**** Memory Tracer closed ****")

    def start_train(self, chunk_size):
        self.chunk_size = chunk_size
        if self.use_async_mem_monitor:
            self.async_mem_monitor.start()
        log_dist("**** Memory Tracer started ****")

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
            self.memory_infos = []
        log_dist("Reset Memory Statistics")

    def trace_memory(self):
        """Record the memory usage of the moment and increase moment counter."""
        if torch.distributed.is_initialized():
            rank = self.local_rank
        else:
            rank = 0
        gpu_device = torch.device(f"cuda:{rank}")
        cpu_device = torch.device("cpu:0")
        gpu_used = get_sys_memory_used(gpu_device)

        if self.metronome.is_warmup:
            # Get peak memory between cur tracing and the prev tracing
            if self.use_async_mem_monitor:
                max_mem_period = self.async_mem_monitor.finish()
                gpu_used = max(max_mem_period, gpu_used)
                self.async_mem_monitor.start()

            cpu_used = get_sys_memory_used(cpu_device)
            self.memory_infos.append(
                MemoryInfo(
                    cpu_total=cpu_used,
                    cpu_chunk=self.cpu_chunk_used_mem,
                    # detected cpu memory usage (already excluded pinned memory) - chunk non
                    # pinned memory usage = system cpu usage (non-chunk cpu memory)
                    cpu_sys=cpu_used
                    - (self.cpu_chunk_used_mem - self.cpu_chunk_used_mem_pinned),
                    gpu_total=gpu_used,
                    gpu_chunk=self.gpu_chunk_used_mem,
                    gpu_sys=gpu_used - self.gpu_chunk_used_mem,
                )
            )
            cur_mom = self.metronome.moment
            assert len(self.memory_infos) - 1 == cur_mom

        self.metronome.tiktac()

    def add(self, device_type: str, size_in_bytes: int, is_pinned: bool = False):
        if device_type == "cpu":
            self.cpu_chunk_used_mem += size_in_bytes
            if is_pinned:
                self.cpu_chunk_used_mem_pinned += size_in_bytes
        elif device_type == "cuda":
            self.gpu_chunk_used_mem += size_in_bytes
        else:
            raise f"device type {device_type} is not supported"

    def delete(self, device_type, size_in_bytes, is_pinned: bool = False):
        if device_type == "cpu":
            self.cpu_chunk_used_mem -= size_in_bytes
            if is_pinned:
                self.cpu_chunk_used_mem_pinned -= size_in_bytes
        elif device_type == "cuda":
            self.gpu_chunk_used_mem -= size_in_bytes
        else:
            raise f"device type {device_type} is not supported"

    def remaining_chunk_mem(self, device_type):
        """
        Return the remainig chunkable memory on device_type,
        which can be used to host chunks.
        """
        size = self.available_chunk_mem(device_type) - self.used_chunk_mem(device_type)
        logger.debug(
            f"remaining_chunk_mem on {device_type} {size / 1e6} MB on mement {self.metronome.moment}"
        )
        return size

    def used_chunk_mem(self, device_type):
        """
        Return the used chunkable memory on device_type.
        """
        if device_type == "cpu":
            return self.cpu_chunk_used_mem
        elif device_type == "cuda":
            return self.gpu_chunk_used_mem
        else:
            raise RuntimeError(f"unknown device type {device_type}")

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
        if self.metronome.training_stage != TrainingStage.UNSTART:
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
                if self.metronome.training_stage == TrainingStage.ADAM:
                    # There is no activation during Adam stage, so we can use all the GPU
                    # mem for chunks. Need 2 * chunk_size for buffer, save 6 here for now.
                    ava_mem = self.overall_gpu_mem - 4 * self.chunk_size * 4
                    logger.debug(f"GPU available_chunk_mem is {ava_mem / 1e6} MB")
                    return ava_mem
                else:
                    # TODO(jiaruifang) using a guessed number -- 1/3 of the GPU
                    # mem is used for chunk.
                    return self.overall_gpu_mem * self.warmup_gpu_chunk_mem_ratio

        if device_type == "cpu":
            local_world_size = get_local_world_size()
            if self.metronome.training_stage != TrainingStage.ADAM:
                return self.overall_cpu_mem - self.max_cpu_sys_used / local_world_size
            else:
                return self.overall_cpu_mem
        elif device_type == "cuda":
            msc_factor = get_world_size()
            if self.metronome.training_stage == TrainingStage.ADAM:
                return self.overall_gpu_mem - 4 * self.chunk_size * 4
            elif self.metronome.training_stage == TrainingStage.FWD:
                next_mom = self.metronome.next_moment()
                cur_mom = self.metronome.moment
                next_mom_ava_mem = (
                    self.overall_gpu_mem - self.memory_infos[next_mom].gpu_sys
                )
                cur_mom_ava_mem = (
                    self.overall_gpu_mem - self.memory_infos[cur_mom].gpu_sys
                )
                return (
                    min(next_mom_ava_mem, cur_mom_ava_mem)
                    - msc_factor * 2 * self.chunk_size
                )
            elif self.metronome.training_stage == TrainingStage.BWD:
                next_mom = self.metronome.next_moment()
                cur_mom = self.metronome.moment
                next_mom_ava_mem = (
                    self.overall_gpu_mem - self.memory_infos[next_mom].gpu_sys
                )
                cur_mom_ava_mem = (
                    self.overall_gpu_mem - self.memory_infos[cur_mom].gpu_sys
                )
                return (
                    min(next_mom_ava_mem, cur_mom_ava_mem)
                    - msc_factor * 2 * self.chunk_size * msc_factor
                )
