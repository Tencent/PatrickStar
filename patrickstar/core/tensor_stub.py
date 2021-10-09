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

from patrickstar.core.const import AccessType, ParamType


class TensorInfo(object):
    r"""The info related to certain tensor."""

    def __init__(
        self,
        chunk_id: int,
        tensor_id: int,
        start_offset: int,
        numel: int,
        param: torch.nn.Parameter,
        access_type: AccessType,
        param_name="",
    ):
        self.tensor_id = tensor_id
        self.chunk_id = chunk_id
        self.start_offset = start_offset
        self.numel = numel
        self.param = param
        self.tensor_name = (
            f"{param_name}.data"
            if (access_type == AccessType.DATA)
            else f"{param_name}.grad"
        )
        self.access_type = access_type

    def __str__(self):
        return (
            f"tensor_id: {self.tensor_id}, name: {self.tensor_name}, "
            f"shape: {self.param.shape}, chunk_id: {self.chunk_id}, "
            f"start_offset: {self.start_offset}, numel: {self.numel}, status: {self.status()}"
        )

    def status(self):
        if self.param.ps_attr.param_type == ParamType.TORCH_BASED:
            return None
        else:
            return self.param.ps_attr.get_status(self.access_type)
