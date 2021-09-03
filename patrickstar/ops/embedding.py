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

import torch
import torch.nn as nn
from patrickstar.utils import logger
from patrickstar.core import register_param, ParamType

class _CopyInputToCPU(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        return input_.to(torch.device('cpu:0'))

    @staticmethod
    def forward(ctx, input_):
        logger.info(
            f'Entrying CPU Emedding FWD, copy input to cpu and {input_.dtype}')
        return input_.to(torch.device('cpu:0'))

    @staticmethod
    def backward(ctx, grad_output):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        logger.info(
            'Entrying CPU Emedding BWD, copy grad_output to cuda, fp32->fp16')
        return grad_output.to(target_device)


class _CopyActToGPU(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')

        return input_.to(target_device)

    @staticmethod
    def forward(ctx, input_):
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')

        logger.info(
            f'Entrying CPU Emedding BWD, copy grad_output to cuda, input dtype {input_.dtype}'
        )
        return input_.to(target_device)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(torch.device('cpu:0')).float()


def copy_to_cpu(input_):
    return _CopyInputToCPU.apply(input_)


def copy_to_gpu(input_):
    return _CopyActToGPU.apply(input_)


class Embedding(nn.Embedding):
    use_cpu = False
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Embedding.instances.append(self)

    def forward(self, input_):
        if Embedding.use_cpu:
            input_ = copy_to_cpu(input_)
        else:
            input_ = copy_to_gpu(input_)
        output = super().forward(input_)
        if Embedding.use_cpu:
            output = copy_to_gpu(output)
        return output.to(torch.half)
