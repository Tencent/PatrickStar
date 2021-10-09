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

from .const import TensorStatus, AccessType, ParamType


class PSTensor(object):
    global_id = 0

    def __init__(self):
        self.tensor = None
        self.id = PSTensor.global_id
        self.status = TensorStatus.FREE
        PSTensor.global_id += 1

    def __str__(self):
        return f"id: {self.id}, status: {self.status}, tensor: {self.tensor}"


class PSParameter(object):
    r""""""

    def __init__(
        self,
        param: torch.nn.Parameter,
        param_type: ParamType,
        data_type: torch.dtype,
        name: str = None,
    ):
        """
        PSParameter can have different dtype compare to param.

        Args
            param: :class:`torch.nn.Parameter`. PSParameter will manage its data and grad.
            param_type: :class:`ParamType`. The torch based param is managed by pytorch,
                while chunk_based is managed by patrickstar chunks.
            data_type: :class:`torch.dtype`. Dtype of PSParameter, can be different from
                `param`.
            name: str
        """
        self.name = name

        self.numel = param.numel()
        self.shape = param.shape

        self.data_type = data_type
        self.param_type = param_type

        self.data_chunk_id = None
        self.grad_chunk_id = None

        if self.param_type == ParamType.CHUNK_BASED:
            self.data_tensor = PSTensor()
            if param.requires_grad:
                self.grad_tensor = PSTensor()
            else:
                self.grad_tensor = None

        # Whether the param belongs to local chunk.
        self._is_local = True

    def __str__(self):
        return (
            f"name: {self.name}, numel: {self.numel}, shape: {self.shape}, "
            f"data_type: {self.data_type}, param_type: {self.param_type}, "
            f"is_local: {self.is_local()}"
        )

    def is_local(self):
        return self._is_local

    def reset_shape(self, new_shape):
        self.shape = new_shape
        self.numel = new_shape.numel()

    def data_id(self):
        return self.get_tensor_id(AccessType.DATA)

    def grad_id(self):
        return self.get_tensor_id(AccessType.GRAD)

    def _access_ps_tensor(self, access_type: AccessType):
        if self.param_type != ParamType.CHUNK_BASED:
            raise ValueError
        if not isinstance(access_type, AccessType):
            raise ValueError
        if access_type == AccessType.DATA:
            return self.data_tensor
        elif access_type == AccessType.GRAD:
            return self.grad_tensor

    def get_tensor_id(self, access_type: AccessType):
        """
        Get the tensor id of chunk based tensor.
        For torch based tensor, return -1.
        """
        if self.param_type == ParamType.TORCH_BASED:
            return -1
        else:
            return self._access_ps_tensor(access_type).id

    def set_tensor(self, tensor: torch.Tensor, access_type: AccessType):
        ps_tensor = self._access_ps_tensor(access_type)
        ps_tensor.tensor = tensor.view(self.shape)

    def access_tensor(self, access_type: AccessType):
        return self._access_ps_tensor(access_type).tensor

    def get_status(self, access_type: AccessType):
        return self._access_ps_tensor(access_type).status

    def set_status(self, status: TensorStatus, access_type: AccessType):
        """
        Only in COMPUTE status when tensor will point to chunk payload.
        Otherwise, the tensor should be None to prevent unnecessary copy.
        TODO(jiaruifang) Need to param reset data和grad
        """
        ps_tensor = self._access_ps_tensor(access_type)
        ps_tensor.status = status
        if status != TensorStatus.COMPUTE:
            ps_tensor.tensor = None


def register_param(param, param_type, data_type, name=None):
    assert isinstance(param, torch.nn.Parameter)
    if not hasattr(param, "ps_attr"):
        param.ps_attr = PSParameter(param, param_type, data_type, name)


def is_param_registered(param) -> bool:
    assert isinstance(param, torch.nn.Parameter)
    return hasattr(param, "ps_attr")
