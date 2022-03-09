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

from enum import Enum


class ChunkState(Enum):
    r"""Chunk state during training."""
    RELEASED = 0  # empty remote enter
    COMPUTE = 1  # in computation, cannot be moved.
    HOLD = 2  # can be moved or be released.


class TensorState(Enum):
    r"""Tensor state during training

    Notice that this is the state of the tensor in the chunk,
    while `ChunkState` is the state of the whole state.
    """
    RELEASED = 0  # empty remote enter
    COMPUTE = 1  # in computation, cannot be moved.
    HOLD = 2  # can be moved or be released.


class ParamType(Enum):
    CHUNK_BASED = 0
    TORCH_BASED = 1
