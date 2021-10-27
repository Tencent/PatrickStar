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

from collections import namedtuple

import psutil


ps_mem_info = namedtuple("ps_mem_info", ["total", "free", "cached", "buffers", "used"])


def get_memory_info():
    try:
        # psutil reads the memory info from /proc/memory_info,
        # which results in returning the host memory instead of
        # that of container.
        # Here we try to read the container memory with method in:
        # https://stackoverflow.com/a/46213331/5163915
        # TODO(zilinzhu) Make this robust on most OS.
        mems = {}
        with open("/sys/fs/cgroup/memory/memory.meminfo", "rb") as f:
            for line in f:
                fields = line.split()
                mems[fields[0]] = int(fields[1]) * 1024
        total = mems[b"MemTotal:"]
        free = mems[b"MemFree:"]
        cached = mems[b"Cached:"]
        buffers = mems[b"Buffers:"]
        used = total - free - cached - buffers
        if used < 0:
            used = total - free
        mem_info = ps_mem_info(
            total=total, free=free, cached=cached, buffers=buffers, used=used
        )
    except FileNotFoundError:
        mems = psutil.virtual_memory()
        mem_info = ps_mem_info(
            total=mems.total,
            free=mems.free,
            cached=mems.cached,
            buffers=mems.buffers,
            used=mems.used,
        )
    return mem_info
