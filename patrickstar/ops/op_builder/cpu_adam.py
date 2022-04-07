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

"""
Copyright 2020 The Microsoft DeepSpeed Team
"""
import os
import sys

from .builder import CUDAOpBuilder


class CPUAdamBuilder(CUDAOpBuilder):
    BUILD_VAR = "DS_BUILD_CPU_ADAM"
    NAME = "cpu_adam"
    BASE_DIR = "patrickstar/ops/csrc"

    def __init__(self):
        super().__init__(name=self.NAME)

    def is_compatible(self):
        # Disable on Windows.
        return sys.platform != "win32"

    def absolute_name(self):
        return f"patrickstar.ops.adam.{self.NAME}_op"

    def sources(self):
        return [
            os.path.join(CPUAdamBuilder.BASE_DIR, "adam/cpu_adam.cpp"),
        ]

    def include_paths(self):
        import torch

        cuda_include = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "include")
        return [os.path.join(CPUAdamBuilder.BASE_DIR, "includes"), cuda_include]

    def cxx_args(self):
        import torch

        cuda_lib64 = os.path.join(torch.utils.cpp_extension.CUDA_HOME, "lib64")
        cpu_arch = self.cpu_arch()
        simd_width = self.simd_width()

        return [
            "-O3",
            "-std=c++14",
            f"-L{cuda_lib64}",
            "-lcudart",
            "-lcublas",
            "-g",
            "-Wno-reorder",
            cpu_arch,
            "-fopenmp",
            simd_width,
        ]
