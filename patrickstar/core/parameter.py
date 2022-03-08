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

from .const import TensorState, ParamType


class TensorInfo:
    r"""The info related to certain tensor."""

    def __init__(
        self,
        chunk_id: int,
        param: torch.nn.Parameter,
        start_offset: int,
    ):
        self.chunk_id = chunk_id
        self.param = param
        self.start_offset = start_offset


class PSParameter:
    global_id = 0

    def __init__(
        self,
        param: torch.nn.Parameter,
        param_type: ParamType,
        dtype: torch.dtype,
        name: str = None,
    ):
        """
        PSParameter can have different dtype compare to param.

        Args
            param: :class:`torch.nn.Parameter`. PSParameter will manage its data and grad.
            param_type: :class:`ParamType`. The torch based param is managed by pytorch,
                while chunk_based is managed by patrickstar chunks.
            dtype: :class:`torch.dtype`. Dtype of PSParameter, can be different from
                `param`.
            name: str
        """
        self.name = name

        self.numel = param.numel()
        self.shape = param.shape

        self.dtype = dtype
        self.param_type = param_type

        self.info = None
        self.state = TensorState.RELEASED

        # Whether the param belongs to local chunk.
        self._is_local = True

    def is_local(self):
        return self._is_local

    def is_chunk_based(self):
        return self.param_type == ParamType.CHUNK_BASED

    def is_torch_based(self):
        return self.param_type == ParamType.TORCH_BASED


def register_param(param, param_type, name=None):
    assert isinstance(param, torch.nn.Parameter)
    if param_type == ParamType.CHUNK_BASED and param.dtype != torch.float:
        raise RuntimeError("only support float parameter in chunk")
    if not hasattr(param, "ps_attr"):
        param.ps_attr = PSParameter(param, param_type, param.dtype, name)


def is_registered(param) -> bool:
    assert isinstance(param, torch.nn.Parameter)
    return hasattr(param, "ps_attr")
