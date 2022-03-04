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

from patrickstar.core.comm import CommInfo
from patrickstar.core.const import ChunkState
from patrickstar.core.memtracer import RuntimeMemTracer
from patrickstar.core.parameter import TensorInfo
from patrickstar.utils import get_rank, getsizeof


class Chunk:
    def __init__(
        self,
        capacity: int,
        chunk_id: int,
        memory_tracer: RuntimeMemTracer,
        local_rank: int = 0,
        is_dummy: bool = False,
    ):
        r"""
        Chunk is the minimal unit of the data transfer.
        It is a contiguous memory for saving tensors.

        Chunk does no know if we are doing distributed training or not.
        Every process will observe its own chunk instances.

        Args:
            capacity: int. The maximum number of elements in the chunk.
            chunk_id: int.
            local_rank: int.
            is_dummy: bool.
        """
        self.chunk_id = chunk_id
        self.comm_info = CommInfo(chunk_id=chunk_id)
        # payload numel does not equal to capacity. payload can be None.
        self.capacity = capacity
        self.local_rank = local_rank
        self._is_dummy = is_dummy
        self.memory_tracer = memory_tracer

        self.payload = None
        self._pin_flag = False

        self.end_pos = 0
        self.params = []
        # the number of params in compute state
        self.num_in_compute = 0

    def is_dummy(self):
        return self._is_dummy

    def is_local(self):
        return get_rank() == self.comm_info.offset

    def get_chunk_space(self):
        r"""Size of the chunk (Bytes)."""
        return getsizeof(torch.float) * self.capacity

    def get_payload_space(self):
        r"""Size of the payload (Bytes)."""
        if self.payload is None:
            return 0
        else:
            return getsizeof(self.payload.dtype) * self.payload.numel()

    def get_device(self):
        r"""Get device of the payload of chunk, return None if not allocated."""
        if self.payload is not None:
            return self.payload.device
        else:
            return None

    def get_state(self):
        r"""
        When payload is None, the state is `RELEASED`,
        otherwise, state of the chunk is decided by its tensors.
        """
        if self.payload is None:
            return ChunkState.RELEASED

        # Distributed training need to fix the chunk on the compute device.
        if self.num_in_compute > 0:
            return ChunkState.COMPUTE
        else:
            return ChunkState.HOLD

    def pin(self):
        self._pin_flag = True

    def unpin(self):
        self._pin_flag = False

    def is_pin(self):
        return self._pin_flag

    def can_fit(self, numel):
        return self.capacity - self.end_pos >= numel

    def add_param(self, param):
        assert param.dtype == torch.float
        numel = param.ps_attr.numel
        if not self.can_fit(numel):
            return False
        self.params.append(param)
        param.ps_attr.info = TensorInfo(self.chunk_id, param, self.end_pos)
        self.end_pos += numel
        return True

    def allocate_payload(self, device):
        r"""Allocate payload on device for the chunk."""
        self.payload = torch.zeros(
            self.capacity,
            dtype=torch.float,
            device=device,
            pin_memory=(device.type == "cpu"),
        )
        self.memory_tracer.add(
            device.type,
            self.get_payload_space(),
            self.payload.is_pinned(),
        )
        self.state = ChunkState.HOLD

    def release_payload(self):
        r"""Release the payload."""
        self.memory_tracer.delete(
            self.get_device().type,
            self.get_payload_space(),
            self.payload.is_pinned(),
        )
        del self.payload
        self.payload = None
        self.state = ChunkState.RELEASED

    def move(self, target_device: torch.device):
        r"""
        Move the chunk to `target_device` synchronizely.
        NOTE() Please check if the `target_device` has enough room before.

        Args:
            target_device: :class:`torch.device`.
        """
        src_device = self.get_device()
        assert src_device is not None and src_device != target_device

        if target_device.type == "cpu":
            self.payload = torch.empty(
                self.payload.shape,
                dtype=self.payload.dtype,
                device="cpu:0",
                pin_memory=True,
            )
            self.payload.copy_(self.payload)
        elif target_device.type == "cuda":
            self.payload = self.payload.pin_memory()
            self.payload = self.payload.to(target_device)

        self.memory_tracer.delete(
            src_device.type,
            self.get_payload_space(),
            self.payload.is_pinned(),
        )
        self.memory_tracer.add(
            target_device.type,
            self.get_payload_space(),
            self.payload.is_pinned(),
        )
