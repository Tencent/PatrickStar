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
from patrickstar.core.memtracer.memtracer import RuntimeMemTracer
from patrickstar.utils.helper import getsizeof


class MemoryCache(object):
    def __init__(self, capacity, memtracer: RuntimeMemTracer):
        r""" "
        A cache of chunk to avoid too much memory allocation and free.
        `capacity` chunks always stay in the GPU memory.
        If we have allocated a chunk on the target device, just reuse the cached one.
        Params:
            `capacity` : the capacity size of each type of tensor cache list.
        Returns:
            None or a `torch.Tensor`.
        """
        self._capacity = capacity
        self._cached_tensors = {}
        self._memtracer = memtracer

    def _new_mem(self, size, data_type, device_type, pin_memory):
        space_size = getsizeof(data_type)
        ret = torch.zeros(
            size,
            dtype=data_type,
            device=device_type,
            pin_memory=pin_memory,
        )
        self._memtracer.add(device_type.type, space_size)
        return ret

    def pop_or_allocate(
        self,
        device_type: torch.device,
        size: int,
        data_type: torch.dtype,
        pin_memory: bool,
    ) -> torch.Tensor:
        """
        Return a tensor including `size` `device_type` elements on `device_type`.
        Delete the reference to the tenor in MemoryCache.
        Return:
            torch.Tensor
        """
        assert isinstance(
            device_type, torch.device
        ), "device_type must be type of torch.device"
        if (device_type, data_type) not in self._cached_tensors:
            return self._new_mem(size, data_type, device_type, pin_memory)
        tensors = self._cached_tensors[(device_type, data_type)]
        i = -1
        for i in range(len(tensors)):
            if tensors[i].numel() == size:
                break
        if i == -1:
            return self._new_mem(size, data_type, device_type, pin_memory)
        new_tensor_ref = tensors[i]
        # delete the reference to tensors[i] in MemoryCache
        tensors.pop(i)
        return new_tensor_ref

    def push(self, payload):
        """
        NOTE() must set payload to None outside of this function.
        Recycle a payload tensor.
        If the cache is fulled, delete the payload.
        Returns:
            success pushed or not.
        """
        device_type = payload.device
        data_type = payload.dtype
        if (device_type, data_type) not in self._cached_tensors and self._capacity > 0:
            self._cached_tensors[(device_type, data_type)] = [payload.zero_()]
        else:
            # the cache is fulled
            if len(self._cached_tensors[(device_type, data_type)]) == self._capacity:
                del payload
                size = payload.numel()
                space_size = getsizeof(data_type) * size
                self._memtracer.delete(device_type.type, space_size)
            else:
                self._cached_tensors[(device_type, data_type)].append(payload.zero_())
