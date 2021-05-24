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
    COMPUTE = 1
    # Chunk内存被分配出来，持有有意义的数据
    HOLD = 2
    # Chunk内存被分配出来，持有无意义数据
    FREE = 3
    # Chunk内存没有被分配出来
    RELEASED = 4


# 数据在计算逻辑中的状态
class PSTensorStatus(Enum):
    # 正在被用于计算，不能随意迁移
    COMPUTE = 0
    # 可以迁移，不能释放
    HOLD = 1
    # 可以释放
    FREE = 2
    UNINIT = 3


# chunk的位置
class PSChunkLocStatus(Enum):
    CPU_PART = 0
    GPU_PART = 1
    GPU_DUP = 2
    UNINIT = 3
    # 调试使用，真实场景不存在这种状态
    CPU_DUP = 4
