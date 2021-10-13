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

import pickle

import fire
import matplotlib.pyplot as plt


def visualize_memory(dict):
    gpu_memory = dict["gpu_memory"]
    gpu_memory = [m / 1024 / 1024 for m in gpu_memory]
    time_stamps = dict["timestamp"]
    step_start = dict["step_start"]
    step_end = dict["step_end"]

    time_stamps = [t - step_start[0] for t in time_stamps]
    step_end = [t - step_start[0] for t in step_end]
    step_start = [t - step_start[0] for t in step_start]

    # plot figure
    plt.style.use("ggplot")

    plt.plot(time_stamps, gpu_memory)

    for t in step_start:
        plt.axvline(x=t, c="g")
    for t in step_end:
        plt.axvline(x=t, c="b")

    plt.xlabel("time/s")
    plt.ylabel("memory/MB")

    plt.title("GPU memory by time")

    plt.show()


def visualize_profile(filename):
    # load profile data
    with open(filename, "rb") as f:
        dict = pickle.load(f)

    visualize_memory(dict)


if __name__ == "__main__":
    fire.Fire(visualize_profile)