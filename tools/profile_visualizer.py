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
import sys

import fire
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_memory(dict, memory_type="GPU"):
    memory_type = memory_type.upper()
    if memory_type not in ["CPU", "GPU"]:
        raise ValueError(f"memory_type {memory_type} not supported.")

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


def visualize_access(dict):
    raw_access_info = dict["chunk_life_cycle"]

    start_time = sys.float_info.max
    for chunk_id, chunk_access_info in raw_access_info.items():
        if len(chunk_access_info["life_cycle"]) == 0:
            continue
        start_time = min(start_time, chunk_access_info["life_cycle"][0][0])

    # TODO(zilinzhu) Currently the chunk id is not correspond to
    # the index in acess_info.
    access_info = {}
    for chunk_id, chunk_access_info in raw_access_info.items():
        chunk_access_info = raw_access_info[chunk_id]
        chunk_type = chunk_access_info["type"]
        raw_life_cycle = chunk_access_info["life_cycle"]
        # Do not show the empty chunk of remote optimizer states.
        if len(raw_life_cycle) == 0:
            continue
        if chunk_type not in access_info:
            access_info[chunk_type] = []
        life_cycle = [(data[0] - start_time, data[2]) for data in raw_life_cycle]
        if life_cycle is not None:
            access_info[chunk_type].append(life_cycle)

    plt.style.use('ggplot')
    _, axis = plt.subplots()

    end_time = dict["end_time"] - start_time
    offset = 0
    for chunk_type, type_access_info in access_info.items():
        for i in range(len(type_access_info)):
            chunk_access_info = type_access_info[i]
            for j in range(len(chunk_access_info)):
                timestamp, device = chunk_access_info[j]
                if j + 1 < len(chunk_access_info):
                    next_timestamp, _ = chunk_access_info[j + 1]
                else:
                    next_timestamp = end_time
                if device is None:
                    color = "#fff"
                elif device.type == "cpu":
                    color = "#e9616c"
                else:
                    color = "#3385fe"
                rect = patches.Rectangle(
                    (timestamp, i + offset + 1), next_timestamp - timestamp, 1, color=color,
                    alpha=1 if device is not None else 0)
                axis.add_patch(rect)
        offset += len(type_access_info)

    axis.set_xlim([0, end_time])
    axis.set_ylim([0, offset])

    plt.xlabel("time/s")
    plt.ylabel("fp16; fp32; momentum; variance")
    plt.title("Chunk location by time")

    plt.show()


def visualize_profile(filename, fig_type="memory", memory_type="GPU"):
    # load profile data
    with open(filename, "rb") as f:
        dict = pickle.load(f)

    if fig_type == "memory":
        visualize_memory(dict, memory_type=memory_type)
    elif fig_type == "access":
        visualize_access(dict)
    else:
        raise ValueError(f"fig_type {fig_type} not supported.")


if __name__ == "__main__":
    fire.Fire(visualize_profile)
