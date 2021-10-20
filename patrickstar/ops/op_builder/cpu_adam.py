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
