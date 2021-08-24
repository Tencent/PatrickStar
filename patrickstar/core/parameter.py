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
from .const import PSTensorStatus, AccessType
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
        self.grad_chunk_id = None

        self.data_tensor = PSTensor()
        if param.requires_grad:
            self.grad_tensor = PSTensor()
        else:
            self.grad_tensor = None

        # 参数是否属于进程的本地Chunk
        self._is_local = True
        # 参数是否交给Torch做内存管理，而不是Chunk
        self._is_torch = False

    def is_local(self):
        return self._is_local

    def reset_shape(self, new_shape):
        self.shape = new_shape
        self.numel = new_shape.numel()

    def data_id(self):
        return self.get_tensor_id(AccessType.DATA)

    def grad_id(self):
        return self.get_tensor_id(AccessType.GRAD)

    def get_tensor_id(self, access_type: AccessType):
        if access_type == AccessType.DATA:
            return self.data_tensor.id
        elif access_type == AccessType.GRAD:
            return self.grad_tensor.id
        else:
            raise ValueError

    def set_tensor(self, tensor: torch.Tensor, access_type: AccessType):
        if access_type == AccessType.DATA:
            self.data_tensor.tensor = tensor.view(self.shape)
        elif access_type == AccessType.GRAD:
            self.grad_tensor.tensor = tensor.view(self.shape)
        else:
            raise ValueError

    def access_tensor(self, access_type: AccessType):
        if access_type == AccessType.DATA:
            return self.data_tensor.tensor
        elif access_type == AccessType.GRAD:
            if self._is_torch:
                raise RuntimeError
            return self.grad_tensor.tensor
        else:
            raise ValueError

    def get_status(self, access_type: AccessType):
        if access_type == AccessType.DATA:
            return self.data_tensor.status
        elif access_type == AccessType.GRAD:
            return self.grad_tensor.status
        else:
            raise ValueError

    def set_status(self, status: PSTensorStatus, access_type: AccessType):
        """
        只有COMPUTE状态tensor指向chunk payload
        HOLD和FREE状态都悬空
        TODO(jiaruifang)还需要param reset data和grad
        """
        if access_type == AccessType.DATA:
            self.data_tensor.status = status
            if status != PSTensorStatus.COMPUTE:
                self.data_tensor.tensor = None
        elif access_type == AccessType.GRAD:
            self.grad_tensor.status = status
            if status != PSTensorStatus.COMPUTE:
                self.grad_tensor.tensor = None
        else:
            raise ValueError(
                f'set status {status} when access type is {access_type}')


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
    else:
        raise RuntimeError(
            "Cannot both register_torch_param and register_param")


def is_torch_param(param):
    assert isinstance(param, torch.nn.Parameter)
    if hasattr(param, 'ps_attr'):
        return param.ps_attr._is_torch
    else:
        return False
