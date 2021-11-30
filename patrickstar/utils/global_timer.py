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

from .logging import logger
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
        logger.info("------------- PROFILE RESULTS ----------------")
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


my_timer = GlobalTimer()


class DataMoveCnter(metaclass=SingletonMeta):
    def __init__(self):
        self.amount_dict = {}
        self.times_dict = {}

    def update(self, key_name, tensor_size):
        my_timer = GlobalTimer()
        if not my_timer.start_flag:
            return
        if key_name in self.times_dict:
            self.times_dict[key_name] += 1
            self.amount_dict[key_name] += tensor_size
        else:
            self.times_dict[key_name] = 1
            self.amount_dict[key_name] = tensor_size

    def reset(self):
        for k, _ in self.times_dict.items():
            self.times_dict[k] = 0
            self.amount_dict[k] = 0

    def print(self):
        print("------------- DATA MOVE RESULTS --------------")
        my_timer = GlobalTimer()
        for k, v in self.times_dict.items():
            bwd = 0
            if k in my_timer.elapse_stat and self.amount_dict[k] != 0:
                bwd = self.amount_dict[k] / my_timer.elapse_stat[k]
                print(
                    f"{k}: {self.amount_dict[k] / 1024 / 1024} MB, {v} times, {bwd / 1024 / 1024} MB/s"
                )
            else:
                print(f"{k}: {self.amount_dict[k] / 1024 / 1024} MB")


data_move_cnter = DataMoveCnter()
