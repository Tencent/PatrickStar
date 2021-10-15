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

import pickle
import time

from patrickstar.utils import SingletonMeta


class Profiler(metaclass=SingletonMeta):
    def __init__(self):
        self._nested_level = 0
        self.start_time = None
        self.warmup_finish_time = None
        self.end_time = None
        # memory info
        # [(moment, time, memory)]
        self.gpu_memory_used = []
        self.gpu_chunk_memory_used = []
        self.cpu_memory_used = []
        self.cpu_chunk_memory_used = []
        # stage info
        # [(time, stage_converted)]
        self.stage_convert_time = []
        # chunk info
        # {chunk_id:
        #     "type": type,
        #     "life_cycle": [(time, type, to_device)]}
        self.chunk_life_cycle = {}

    def start(self):
        if self.start_time is None:
            self.start_time = time.time()
        self._nested_level += 1

    def end(self):
        self._nested_level = max(0, self._nested_level - 1)
        if self._nested_level == 0:
            self.end_time = time.time()

    def started(self):
        return self._nested_level > 0
    
    def warmup_finish(self):
        if self.warmup_finish_time is None:
             self.warmup_finish_time = time.time()

    def state_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time if self.end_time is not None else time.time(),
	    "warmup_finish_time" : self.warmup_finish_time,
            "gpu_memory_used": self.gpu_memory_used,
            "gpu_chunk_memory_used": self.gpu_chunk_memory_used,
            "cpu_memory_used": self.cpu_memory_used,
            "cpu_chunk_memory_used": self.cpu_chunk_memory_used,

            "stage_convert_time": self.stage_convert_time,

            "chunk_life_cycle": self.chunk_life_cycle,
        }

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.state_dict(), f)


profiler = Profiler()
