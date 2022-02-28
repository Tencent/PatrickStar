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

from enum import Enum


class ChunkState(Enum):
    r"""Chunk state during training."""
    FREE = 0
    # Chunk memory is allocated.
    # Tensors are used for computing.
    COMPUTE = 1
    # Holding meaningful data.
    HOLD = 2
    HOLD_AFTER_FWD = 3
    HOLD_AFTER_BWD = 4

    # Chunk memory is not allocated.
    RELEASED = 5


class TensorState(Enum):
    r"""Tensor state during training

    Notice that this is the state of the tensor in the chunk,
    while `ChunkState` is the state of the whole state.
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
