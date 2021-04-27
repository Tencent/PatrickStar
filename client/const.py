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
    # Chunk只在cpu上
    COMPUTE = 1
    HOLD = 2
    FREE = 3


class PSTensorStatus(Enum):
    # 正在被用于计算，不能随意迁移
    COMPUTE = 1
    # 可以迁移，不能释放
    HOLD = 2
    # 可以释放
    FREE = 3
    UNINIT = 4
