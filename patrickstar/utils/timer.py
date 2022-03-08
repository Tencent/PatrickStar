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

# from .logging import logger
from .singleton_meta import SingletonMeta


class GlobalTimer(metaclass=SingletonMeta):
    def __init__(self):
        """
        Timer for different function of the program.
        The naming convention should be {TrainingState}_{function},
        e.g. ADAM_compute
        """
        self.elapse_stat = {}
        self.start_time = {}
        self.start_flag = False

    def start(self):
        self.start_flag = True

    def start_profile(self, key):
        if not self.start_flag:
            return
        if key in self.start_time:
            assert self.start_time[key] == 0, f"Please Check {key} profiling function"
        self.start_time[key] = time.time()

    def finish_profile(self, key):
        if not self.start_flag:
            return
        torch.cuda.current_stream().synchronize()
        if key in self.elapse_stat:
            self.elapse_stat[key] += time.time() - self.start_time[key]
        else:
            self.elapse_stat[key] = time.time() - self.start_time[key]
        self.start_time[key] = 0

    def reset(self):
        if not self.start_flag:
            return
        for k, _ in self.elapse_stat.items():
            self.elapse_stat[k] = 0

    def print(self):
        if not self.start_flag:
            return
        print("------------- PROFILE RESULTS ----------------")
        dot_length = 20
        for k in self.elapse_stat:
            dot_length = max(dot_length, len(k) + 2)
        overall_elapse = (
            self.elapse_stat["FWD"] + self.elapse_stat["BWD"] + self.elapse_stat["ADAM"]
        )
        for k, v in self.elapse_stat.items():
            print(
                f'{k} {"." * (dot_length - len(k))} {v}, {v / overall_elapse * 100} %'
            )
        print(f'TOTAL {"." * (dot_length - len("TOTAL"))} {overall_elapse}')


global_timer = GlobalTimer()
