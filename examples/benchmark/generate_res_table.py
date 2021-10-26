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
import collections
from process_logs import collect_info_from_dir

if __name__ == "__main__":
    overall_res_dict = {}
    overall_file_dict = {}
    for file in os.listdir("./"):
        if os.path.isdir(file) and "logs" in file:
            res_dict, file_dict = collect_info_from_dir(file)
            overall_res_dict.update(res_dict)
            overall_file_dict.update(file_dict)

    detail_res_table = {}
    best_res_table = {}
    pos = {1: 0, 2: 2, 4: 4, 8: 6}

    for k, v in overall_res_dict.items():
        plan = k.split("_")
        model_scale = plan[0]
        bs = plan[1]
        gpu_num = int(plan[2])
        key = (model_scale, bs)
        if key not in detail_res_table:
            detail_res_table[key] = [None for i in range(8)]

        filename = overall_file_dict[k]
        detail_res_table[key][pos[gpu_num]] = v * gpu_num
        detail_res_table[key][pos[gpu_num] + 1] = filename

        if model_scale not in best_res_table:
            best_res_table[model_scale] = [0 for i in range(8)]
        if v * gpu_num > best_res_table[model_scale][pos[gpu_num]]:
            best_res_table[model_scale][pos[gpu_num]] = v * gpu_num
            best_res_table[model_scale][pos[gpu_num] + 1] = bs  # filename

    od = collections.OrderedDict(sorted(detail_res_table.items()))
    with open("benchmark_res.csv", "w") as wfh:
        for k, v in od.items():
            for item in k:
                wfh.write(str(item))
                wfh.write(",")
            for item in v:
                wfh.write(str(item))
                wfh.write(",")
            wfh.write("\n")

    od = collections.OrderedDict(sorted(best_res_table.items()))

    with open("best_res.csv", "w") as wfh:
        for k, v in od.items():
            wfh.write(str(k))
            wfh.write(",")
            for item in v:
                wfh.write(str(item))
                wfh.write(",")
            wfh.write("\n")
