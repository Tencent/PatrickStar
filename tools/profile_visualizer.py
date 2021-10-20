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


import logging
import pickle
import sys

import fire
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def visualize_memory(dict, filename, memory_type="GPU", rm_warmup=False):
    memory_type = memory_type.upper()
    is_torch = False
    if "torch" in filename:
        is_torch = True

    if memory_type not in ["CPU", "GPU"]:
        raise ValueError(f"memory_type {memory_type} not supported.")

    if memory_type == "GPU":
        raw_memory_used = dict["gpu_memory_used"]
        raw_chunk_memory_used = dict["gpu_chunk_memory_used"]
    else:
        raw_memory_used = dict["cpu_memory_used"]
        raw_chunk_memory_used = dict["cpu_chunk_memory_used"]

    warmup_finish_time = dict["warmup_finish_time"]

    raw_stage_convert_time = dict["stage_convert_time"]

    if len(raw_memory_used) == 0:
        logging.warning("Empty profile file.")

    if rm_warmup:
        start_time = warmup_finish_time
    else:
        start_time = raw_memory_used[0][1]

    # moments = [data[0] for data in raw_memory_used]
    time_stamps = [data[1] - start_time for data in raw_memory_used]
    gpu_memory = [data[2] for data in raw_memory_used]
    if raw_chunk_memory_used is None:
        gpu_chunk_memory = [0 for _ in time_stamps]
    gpu_chunk_memory = [data[2] for data in raw_chunk_memory_used]

    gpu_memory = [mem / 1024 / 1024 for mem in gpu_memory]
    gpu_chunk_memory = [mem / 1024 / 1024 for mem in gpu_chunk_memory]

    gpu_non_model_memory = [o - c for o, c in zip(gpu_memory, gpu_chunk_memory)]
    postive_time_stamp = []
    postive_gpu_non_model_memory = []
    for t, m in zip(time_stamps, gpu_non_model_memory):
        if t >= 0.0:
            postive_time_stamp.append(t)
            postive_gpu_non_model_memory.append(m)

    # plot figure
    plt.style.use("ggplot")

    if not is_torch:
        stage_convert_time = [data[0] - start_time for data in raw_stage_convert_time]
        stage_types = [data[1] for data in raw_stage_convert_time]

        for i in range(len(stage_convert_time) - 1):
            start, end = stage_convert_time[i], stage_convert_time[i + 1]
            # Convert Enum to int value so that we don't need to import patrickstar
            # in this script.
            stage = stage_types[i].value
            if stage == 1:
                # TrainStage.FWD
                facecolor = "g"
            elif stage == 2:
                # TrainStage.BWD
                facecolor = "tab:blue"
            elif stage == 3:
                # TrainStage.ADAM
                facecolor = "tab:purple"
            else:
                raise RuntimeError(f"Unexpected stage value: {stage_types[i]}")
            plt.axvspan(start, end, facecolor=facecolor, alpha=0.3)
        # The last Adam stage
        plt.axvspan(
            stage_convert_time[-1], time_stamps[-1], facecolor="tab:purple", alpha=0.2
        )

    plt.plot(time_stamps, gpu_memory, label="total")
    if not is_torch:
        plt.plot(time_stamps, gpu_chunk_memory, label="chunk")
    plt.plot(postive_time_stamp, postive_gpu_non_model_memory, label="non-model")
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

    plt.style.use("ggplot")
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
                    (timestamp, i + offset + 1),
                    next_timestamp - timestamp,
                    1,
                    color=color,
                    alpha=1 if device is not None else 0,
                )
                axis.add_patch(rect)
        offset += len(type_access_info)

    axis.set_xlim([0, end_time])
    axis.set_ylim([0, offset])

    plt.xlabel("time/s")
    plt.ylabel("fp16; fp32; momentum; variance")
    plt.title("Chunk location by time")

    plt.show()


def visualize_profile(filename, fig_type="memory", memory_type="GPU", rm_warmup=True):
    # load profile data
    with open(filename, "rb") as f:
        dict = pickle.load(f)

    if fig_type == "memory":
        visualize_memory(dict, filename, memory_type=memory_type, rm_warmup=rm_warmup)
    elif fig_type == "access":
        visualize_access(dict)
    else:
        raise ValueError(f"fig_type {fig_type} not supported.")


if __name__ == "__main__":
    fire.Fire(visualize_profile)
