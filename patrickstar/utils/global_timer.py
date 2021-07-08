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
class DataMoveCnter(object):
    def __init__(self):
        # self.gpu_to_cpu_times = 0
        # self.gpu_to_cpu_amount = 0
        # self.param_fp32_to_fp16_copy_times = 0
        # self.param_fp32_to_fp16_copy_amount = 0

        # self.cpu_to_gpu_times = 0
        # self.cpu_to_gpu_amount = 0
        # self.grad_fp16_to_fp32_copy_times = 0
        # self.grad_fp16_to_fp32_copy_data_amount = 0

        self.amount_dict = {}
        self.times_dict = {}

    def update(self, key_name, tensor_size):
        if key_name in self.times_dict:
            self.times_dict[key_name] += 1
            self.amount_dict[key_name] += tensor_size
        else:
            self.times_dict[key_name] = 1
            self.amount_dict[key_name] = tensor_size

    def reset(self):
        for k, v in self.times_dict.items():
            self.times_dict[k] = 0
            self.amount_dict[k] = 0

    def print(self):
        logging.info('*********** DATA MOVE RESULTS *************')
        for k, v in self.times_dict.items():
            logging.info(f'{k}: {self.amount_dict[k]/1024/1024} MB, {v} times')
        logging.info('\n\n')


data_move_cnter = DataMoveCnter()
