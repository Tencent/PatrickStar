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
