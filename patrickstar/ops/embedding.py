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

import torch
import torch.nn as nn

from patrickstar.utils import logger


class _CopyInputToCPU(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return input_.to(torch.device("cpu:0"))

    @staticmethod
    def forward(ctx, input_):
        logger.debug(f"Copy input to cpu and {input_.dtype}.")
        return input_.to(torch.device("cpu:0"))

    @staticmethod
    def backward(ctx, grad_output):
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}")
        logger.debug("Copy grad_output to cuda.")
        return grad_output.to(target_device)


class _CopyActToGPU(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}")

        return input_.to(target_device)

    @staticmethod
    def forward(ctx, input_):
        target_device = torch.device(f"cuda:{torch.cuda.current_device()}")

        logger.debug(f"Copy grad_output to cuda, input dtype {input_.dtype}.")
        return input_.to(target_device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(torch.device("cpu:0")).float()


def copy_to_cpu(input_):
    return _CopyInputToCPU.apply(input_)


def copy_to_gpu(input_):
    return _CopyActToGPU.apply(input_)


class Embedding(nn.Embedding):
    r"""CPU Embedding.

    If `use_cpu` is set, the embedding operations will
    be performed on CPU.
    """
    use_cpu = False
    # `instances` is a helper class static member for
    # preprocess context. For detail, see comments there.
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cpu = Embedding.use_cpu
        Embedding.instances.append(self)

    def forward(self, input_):
        if self.use_cpu:
            input_ = copy_to_cpu(input_)
        else:
            input_ = copy_to_gpu(input_)
        output = super().forward(input_)
        if self.use_cpu:
            output = copy_to_gpu(output)
        return output.to(torch.half)
