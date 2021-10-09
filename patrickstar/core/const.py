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


class AccessType(Enum):
    DATA = 1
    GRAD = 2


class ChunkStatus(Enum):
    r"""Chunk status during training."""
    FREE = 0
    # Chunk memory is allocated.
    # Tensors are used for computing.
    COMPUTE = 1
    # Holding meaningful data.
    HOLD = 2
    # Holding meaningless data.
    HOLD_AFTER_FWD = 3
    HOLD_AFTER_BWD = 4

    # Chunk memory is not allocated.
    RELEASED = 5


class TensorStatus(Enum):
    r"""Tensor status during training

    Notice that this is the status of the tensor in the chunk,
    while `ChunkStatus` is the status of the whole status.
    """
    # Can be released.
    FREE = 0
    # In computation, cannot be moved.
    COMPUTE = 1
    # Can be moved, cannot be released.
    HOLD = 2
    HOLD_AFTER_FWD = 3
    HOLD_AFTER_BWD = 4


class TrainingStage(Enum):
    UNSTART = 0
    FWD = 1
    BWD = 2
    ADAM = 3


class ChunkType(Enum):
    PARAM_FP16 = 0
    PARAM_FP32 = 1
    MOMENTUM = 2
    VARIANCE = 3
    UNDEF = 4


class ParamType(Enum):
    CHUNK_BASED = 0
    TORCH_BASED = 1
