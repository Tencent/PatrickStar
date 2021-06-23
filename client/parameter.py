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

global_id = 0


#TODO如何产生全局id？
class PSTensor(object):
    def __init__(self):
        global global_id
        self.ps_tensor = None
        self.ps_id = global_id
        self.ps_status = PSTensorStatus.FREE
        global_id += 1


class PSParameter(object):
    def __init__(self, param: torch.nn.Parameter, name: str = None):
        """
        在torch.nn.Parameter的附加成员变量
        """
        self.ps_name = None
        self.ps_numel = None
        self.ps_shape = None

        self.ps_data_chunk_id = None
        self.ps_grad_chunk_id = None
        # id, status, tensor_data
        self.data_tensor = None
        self.grad_tensor = None

        self.ps_name = name
        self.ps_numel = param.numel()
        self.ps_shape = param.shape

        self.data_tensor = PSTensor()
        if param.requires_grad:
            self.grad_tensor = PSTensor()

        # 参数是否属于进程的本地Chunk
        self._is_local = None
        # 参数是否交给Torch做内存管理，而不是Chunk
        self._is_torch = False

    def is_local(self):
        return self._is_local

    def reset_shape(self, new_shape):
        self.ps_shape = new_shape
        self.ps_numel = new_shape.numel()

    def data_id(self):
        return self.get_tensor_id(AccessType.DATA)

    def grad_id(self):
        return self.get_tensor_id(AccessType.GRAD)

    def get_tensor_id(self, access_type: AccessType):
        if access_type == AccessType.DATA:
            return self.data_tensor.ps_id
        elif access_type == AccessType.GRAD:
            return self.grad_tensor.ps_id
        else:
            raise RuntimeError

    def set_tensor(self, tensor: torch.Tensor, access_type: AccessType):
        if access_type == AccessType.DATA:
            self.data_tensor.ps_tensor = tensor.view(self.ps_shape)
        elif access_type == AccessType.GRAD:
            self.grad_tensor.ps_tensor = tensor.view(self.ps_shape)
        else:
            raise RuntimeError

    def access_tensor(self, access_type: AccessType):
        if access_type == AccessType.DATA:
            return self.data_tensor.ps_tensor
        elif access_type == AccessType.GRAD:
            if self._is_torch:
                raise RuntimeError
            return self.grad_tensor.ps_tensor
        else:
            raise RuntimeError

    def get_status(self, access_type: AccessType):
        if access_type == AccessType.DATA:
            return self.data_tensor.ps_status
        elif access_type == AccessType.GRAD:
            return self.grad_tensor.ps_status
        else:
            raise RuntimeError

    def set_status(self, status: PSTensorStatus, access_type: AccessType):
        """
        只有COMPUTE状态tensor指向chunk payload
        HOLD和FREE状态都悬空
        TODO(jiaruifang)还需要param reset data和grad
        """
        if access_type == AccessType.DATA:
            self.data_tensor.ps_status = status
            if status != PSTensorStatus.COMPUTE:
                self.data_tensor.ps_tensor = None
        elif access_type == AccessType.GRAD:
            self.grad_tensor.ps_status = status
            if status != PSTensorStatus.COMPUTE:
                self.grad_tensor.ps_tensor = None
        else:
            raise RuntimeError(
                f'set status {status} when access type is {access_type}')


def register_param(param, name=None):
    if not hasattr(param, 'ps_attr'):
        param.ps_attr = PSParameter(param, name)


def is_param_registed(param) -> bool:
    return hasattr(param, 'ps_attr')


def register_torch_param(param, name=None):
    if not hasattr(param, 'ps_attr'):
        param.ps_attr = PSParameter(param, name)
        param.ps_attr._is_torch = True
        # param.ps_attr.data_tensor.ps_tensor = param.data


def is_torch_param(param):
    if hasattr(param, 'ps_attr'):
        return param.ps_attr._is_torch
    else:
        return False
