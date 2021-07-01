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

# 统计chunk的lifecycle开关
import logging
import time


class GlobalTimer(object):
    def __init__(self):
        """
        存放时间统计，key命名规则 训练阶段_
        """
        self.elapse_stat = {}

        self.start_time = {}

    def start_profile(self, key):
        if key in self.start_time:
            assert self.start_time[
                key] == 0, f"Please Check {key} profiling function"
        self.start_time[key] = time.time()

    def finish_profile(self, key):
        if key in self.elapse_stat:
            self.elapse_stat[key] += time.time() - self.start_time[key]
        else:
            self.elapse_stat[key] = time.time() - self.start_time[key]
        self.start_time[key] = 0

    def reset(self):
        for k, v in self.elapse_stat.items():
            self.elapse_stat[k] = 0

    def print(self):
        logging.info('*********** PROFILE RESULTS *************')
        for k, v in self.elapse_stat.items():
            logging.info(f'{k}: {v}')


my_timer = GlobalTimer()

# 数据移动
cpu_gpu_move_elapse = 0.
cpu_gpu_move_times = 0
cpu_gpu_move_data_amount = 0

gpu_cpu_move_elapse = 0.
gpu_cpu_move_times = 0
gpu_cpu_move_data_amount = 0
