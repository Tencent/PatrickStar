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
        self.end_time = None
        self.timestamp = []
        self.gpu_memory = []
        self.step_start = []
        self.step_end = []

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

    def state_dict(self):
        return {
            "start_time": self.start_time,
            "end_time": self.end_time if self.end_time is not None else time.time(),
            "timestamp": self.timestamp,
            "gpu_memory": self.gpu_memory,
            "step_start": self.step_start,
            "step_end": self.step_end,
        }

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.state_dict(), f)


torch_profiler = Profiler()
