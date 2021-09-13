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

import logging
import pickle

import fire
import matplotlib.pyplot as plt


def visualize_profile(filename, memory_type="GPU"):
    memory_type = memory_type.upper()
    if memory_type not in ["CPU", "GPU"]:
        raise ValueError(f"memory_type {memory_type} not supported.")

    # load profile data
    with open(filename, "rb") as f:
        dict = pickle.load(f)

    if memory_type == "GPU":
        raw_memory_used = dict["gpu_memory_used"]
        raw_chunk_memory_used = dict["gpu_chunk_memory_used"]
    else:
        raw_memory_used = dict["cpu_memory_used"]
        raw_chunk_memory_used = dict["cpu_chunk_memory_used"]

    raw_stage_convert_time = dict["stage_convert_time"]

    if len(raw_memory_used) == 0:
        logging.warning("Empty profile file.")

    # process profile data
    start_time = raw_memory_used[0][1]

    # moments = [data[0] for data in raw_memory_used]
    time_stamps = [data[1] - start_time for data in raw_memory_used]
    gpu_memory = [data[2] for data in raw_memory_used]
    gpu_chunk_memory = [data[2] for data in raw_chunk_memory_used]

    gpu_memory = [mem / 1024 / 1024 for mem in gpu_memory]
    gpu_chunk_memory = [mem / 1024 / 1024 for mem in gpu_chunk_memory]

    stage_convert_time = [data[0] - start_time for data in raw_stage_convert_time]
    stage_types = [data[1] for data in raw_stage_convert_time]

    # plot figure
    plt.style.use('ggplot')

    for i in range(len(stage_convert_time) - 1):
        start, end = stage_convert_time[i], stage_convert_time[i + 1]
        # Convert Enum to int value so that we don't need to import patrickstar
        # in this script.
        stage = stage_types[i].value
        if stage == 1:
            # TrainStage.FWD
            facecolor = 'g'
        elif stage == 2:
            # TrainStage.BWD
            facecolor = 'tab:blue'
        elif stage == 3:
            # TrainStage.ADAM
            facecolor = 'tab:purple'
        else:
            raise RuntimeError(f"Unexpected stage value: {stage_types[i]}")
        plt.axvspan(start, end, facecolor=facecolor, alpha=0.3)
    # The last Adam stage
    plt.axvspan(stage_convert_time[-1], time_stamps[-1],
                facecolor='tab:purple', alpha=0.2)

    plt.plot(time_stamps, gpu_memory, label="total")
    plt.plot(time_stamps, gpu_chunk_memory, label="chunk")
    plt.legend()

    plt.xlabel("time/s")
    plt.ylabel("memory/MB")

    if memory_type == "GPU":
        plt.title("GPU memory by time")
    else:
        plt.title("CPU memory by time")

    plt.show()


if __name__ == "__main__":
    fire.Fire(visualize_profile)
