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


class PSChunkStatus(Enum):
    # Chunk内存被分配出来，在计算中
    COMPUTE = 0
    # Chunk内存被分配出来，持有有意义的数据
    HOLD = 1
    # Chunk内存被分配出来，持有无意义数据
    HOLD_AFTER_FWD = 2
    HOLD_AFTER_BWD = 3
    FREE = 4
    # Chunk内存没有被分配出来
    RELEASED = 5
    # dist使用，虽然所有tensor都是hold，但被pin在计算设备上


# 数据在计算逻辑中的状态
class PSTensorStatus(Enum):
    # 正在被用于计算，不能随意迁移
    COMPUTE = 0
    # 可以迁移，不能释放
    HOLD = 1
    HOLD_AFTER_FWD = 2
    HOLD_AFTER_BWD = 3
    # 可以释放
    FREE = 4


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
