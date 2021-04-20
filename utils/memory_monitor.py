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
import psutil
import gc
import torch
import logging


def see_memory_usage(message, force=False):
    if not force:
        return
    if torch.distributed.is_initialized(
    ) and not torch.distributed.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    scale = 1024
    scale_name = "KB"
    # Print message except when distributed but not rank 0
    logging.info(message)
    logging.info(
        f"MA {round(torch.cuda.memory_allocated() / scale,2 )} {scale_name} \
        Max_MA {round(torch.cuda.max_memory_allocated() / scale,2)} {scale_name} \
        CA {round(torch.cuda.memory_reserved() / scale,2)} {scale_name} \
        Max_CA {round(torch.cuda.max_memory_reserved() / scale)} {scale_name} "
    )

    vm_stats = psutil.virtual_memory()
    used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
    logging.info(
        f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%'
    )

    # get the peak memory to report correct data, so reset the counter for the next call
    if hasattr(torch.cuda, "reset_peak_memory_stats"):  # pytorch 1.4+
        torch.cuda.reset_peak_memory_stats()
