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

import os
import torch

from .logging import logger


def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return 1


# Use global variable to prevent changing of the environment variable
# and to make sure the warning is only logged once.
_local_world_size = None


def get_local_world_size():
    global _local_world_size
    if _local_world_size is None:
        if torch.distributed.is_initialized():
            if "LOCAL_WORLD_SIZE" in os.environ:
                _local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
            else:
                logger.warning(
                    "If you are training with multiple nodes, it's recommand to "
                    "set LOCAL_WORLD_SIZE manually to make better use of CPU memory. "
                    "Otherwise, get_world_size() is used instead."
                )
                _local_world_size = get_world_size()
        else:
            _local_world_size = 1
    return _local_world_size
