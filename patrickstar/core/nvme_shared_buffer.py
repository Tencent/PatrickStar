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

import torch

from patrickstar.core.chunk_data import Chunk


class NVMeSharedBuffer:
    def __init__(self, num_buffer=2):
        self.buffers = []
        self.num_buffer = num_buffer
        self.next_buffer = 0
        self.buffer_id_to_chunk_id_map = {}

    def get(self, chunk: Chunk):
        if len(self.buffers) == self.next_buffer:
            self.buffers.append(
                torch.zeros(
                    chunk.capacity,
                    dtype=chunk.data_type,
                    device=torch.device("cpu"),
                    pin_memory=True,
                )
            )
        buffer = self.buffers[self.next_buffer]
        assert buffer.numel() == chunk.capacity and buffer.dtype == chunk.data_type
        old_chunk = self.buffer_id_to_chunk_id_map.get(self.next_buffer, None)
        self.buffer_id_to_chunk_id_map[self.next_buffer] = chunk
        self.next_buffer = (self.next_buffer + 1) % self.num_buffer
        return buffer, old_chunk
