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
from .const import PSTensorStatus
import logging


class PSTensor(object):
    global_id = 0

    def __init__(self):
        self.tensor = None
        self.id = PSTensor.global_id
        self.status = PSTensorStatus.FREE
        PSTensor.global_id += 1


class PSParameter(object):
    def __init__(self, param: torch.nn.Parameter, name: str = None):
        """
        在torch.nn.Parameter的附加成员变量
        """
        self.name = name
        self.numel = param.numel()
        self.shape = param.shape

        self.data_chunk_id = None       

        self.data = PSTensor()

        # 参数是否属于进程的本地Chunk
        self._is_local = True
        # 参数是否交给Torch做内存管理，而不是Chunk
        self._is_torch = False

    def is_local(self):
        return self._is_local

    def reset_shape(self, new_shape):
        self.shape = new_shape
        self.numel = new_shape.numel()

    def id(self):
        return self.data.id

    def set_tensor(self, tensor: torch.Tensor):
        self.data.tensor = tensor.view(self.shape)

    def access_tensor(self):
        return self.data.tensor

    def status(self):
        return self.data.status

    def set_status(self, status: PSTensorStatus):
        """
        只有COMPUTE状态tensor指向chunk payload
        HOLD和FREE状态都悬空
        TODO(jiaruifang)还需要param reset data和grad
        """
        self.data.status = status
        if status != PSTensorStatus.COMPUTE:
            self.data.tensor = None


def register_param(param, name=None):
    assert isinstance(param, torch.nn.Parameter)
    if not hasattr(param, 'ps_attr'):
        param.ps_attr = PSParameter(param, name)


def is_param_registed(param) -> bool:
    assert isinstance(param, torch.nn.Parameter)
    return hasattr(param, 'ps_attr')


def register_torch_param(param, name=None):
    assert isinstance(param, torch.nn.Parameter)
    if not hasattr(param, 'ps_attr'):
        param.ps_attr = PSParameter(param, name)
        param.ps_attr._is_torch = True
        # param.ps_attr.data.tensor = param.data


def is_torch_param(param):
    assert isinstance(param, torch.nn.Parameter)
    if hasattr(param, 'ps_attr'):
        return param.ps_attr._is_torch
    else:
        return False
