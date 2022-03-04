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

from .const import TensorState, ParamType


class TensorInfo(object):
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


class PSParameter(object):
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
        if self.param_type == ParamType.CHUNK_BASED:
            self.id = PSParameter.global_id
            PSParameter.global_id += 1
        else:
            self.id = -1

        # Whether the param belongs to local chunk.
        self._is_local = True

    def __str__(self):
        return (
            f"name: {self.name}, numel: {self.numel}, shape: {self.shape}, "
            f"dtype: {self.dtype}, param_type: {self.param_type}, "
            f"is_local: {self.is_local()}"
        )

    def is_local(self):
        return self._is_local

    def reset_shape(self, new_shape):
        self.shape = new_shape
        self.numel = new_shape.numel()


def register_param(param, param_type, name=None):
    assert isinstance(param, torch.nn.Parameter)
    if param_type == ParamType.CHUNK_BASED and param.dtype != torch.float:
        raise RuntimeError("only support float parameter in chunk")
    if not hasattr(param, "ps_attr"):
        param.ps_attr = PSParameter(param, param_type, param.dtype, name)


def is_registered(param) -> bool:
    assert isinstance(param, torch.nn.Parameter)
    return hasattr(param, "ps_attr")
