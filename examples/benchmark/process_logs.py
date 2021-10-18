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

import os
import sys


def is_run_this_file(path, file, res_dict, file_dict):
    """
    collect throughput performance form log
    and update res_dict and file_dict
    ret: is_run, whether to run the script
    """
    model_name = ""
    gpu_num = 0
    bs = 0

    # if the file is not exist.
    # do not execute training
    if not os.path.isfile(path + "/" + file):
        return True

    f = open(path + "/" + file)
    is_run = True
    if not os.path.isdir(file):
        fn_list = file.split(".")[1].split("_")
        for i in range(len(fn_list)):
            if "GPT" in fn_list[i] or "Bert" in fn_list[i]:
                model_name = fn_list[i + 1]
            elif "bs" == fn_list[i]:
                bs = fn_list[i + 1]
            elif "gpu" == fn_list[i]:
                gpu_num = fn_list[i + 1]
        key = model_name + "_" + bs + "_" + gpu_num
        iter_f = iter(f)
        for line in iter_f:
            if "Tflops" in line and "WARM" not in line:
                sline = line.split()
                perf = float(sline[-2])
                if key not in res_dict:
                    res_dict[key] = perf
                    file_dict[key] = file
                else:
                    if res_dict[key] < perf:
                        res_dict[key] = perf
                        file_dict[key] = file
                is_run = False
            if "RuntimeError" in line:
                return False

    return is_run


if __name__ == "__main__":
    res_dict = {}
    file_dict = {}
    if len(sys.argv) > 1:
        PATH = str(sys.argv[1])
    else:
        PATH = "./logs_GPT2small"
    files = os.listdir(PATH)
    for file in files:
        is_run_this_file(PATH, file, res_dict, file_dict)
    # print(res_dict)
    # print(file_dict)

    new_res_list = []
    for k, v in res_dict.items():
        plan = k.split("_")
        new_res_list.append((plan[0], plan[1], plan[2], v, file_dict[k]))

    new_res_list.sort()
    for elem in new_res_list:
        print(elem)
