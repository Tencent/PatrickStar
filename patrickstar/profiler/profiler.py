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
            "warmup_finish_time": self.warmup_finish_time,
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
