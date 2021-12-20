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

import os
import sys
import numpy as np
from scipy.stats import t


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

    perf_list = np.array([])
    if not os.path.isdir(file):
        fn_list = file.split(".")[1].split("_")
        for i in range(len(fn_list)):
            if "gpu" in fn_list[i]:
                model_name = fn_list[i - 1]
                gpu_num = fn_list[i + 1]
            elif "bs" == fn_list[i]:
                bs = fn_list[i + 1]
        key = model_name + "_" + bs + "_" + gpu_num
        iter_f = iter(f)
        for line in iter_f:
            if "Tflops" in line and "WARM" not in line:
                sline = line.split()
                perf = float(sline[-2])

                perf_list = np.append(perf_list, perf)

                is_run = False
            if "RuntimeError" in line:
                return False

    # calculate CI of perf_list
    perf_list = perf_list[1:-1]
    m = perf_list.mean()
    s = perf_list.std()
    dof = len(perf_list) - 1
    confidence = 0.95
    t_crit = np.abs(t.ppf((1 - confidence) / 2, dof))
    ic_perf = (
        -s * t_crit / np.sqrt(len(perf_list)),
        +s * t_crit / np.sqrt(len(perf_list)),
    )

    res_dict[key] = (*ic_perf, m)
    file_dict[key] = file

    return is_run


def collect_info_from_dir(path):
    res_dict = {}
    file_dict = {}
    files = os.listdir(path)
    for file in files:
        is_run_this_file(path, file, res_dict, file_dict)
    print("process ", path)
    return res_dict, file_dict


if __name__ == "__main__":
    res_dict = {}
    file_dict = {}
    if len(sys.argv) > 1:
        PATH = str(sys.argv[1])
    else:
        PATH = "./logs_GPT2small"
    files = os.listdir(PATH)
    res_dict, file_dict = collect_info_from_dir(PATH)
    new_res_list = []
    for k, v in res_dict.items():
        plan = k.split("_")
        # model_name, bs, gpu_num, best perf, file
        new_res_list.append((plan[0], plan[1], plan[2], v, file_dict[k]))

    new_res_list.sort()
    for elem in new_res_list:
        print(elem)
