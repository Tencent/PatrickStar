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


class MemoryCache(object):
    def __init__(self, capacity=2):
        r""" "
        A cache of chunk to avoid too much memory allocation and free.
        `capacity` chunks always stay in the GPU memory.
        If we have allocated a chunk on the target device, just reuse the cached one.
        Params:
            `capacity` : the capacity size of each type of tensor cache list.
        Returns:
            None or a `torch.Tensor`.
        """
        self.capacity_ = capacity
        self.cached_tensors = {}

    def allocate(
        self, device_type: str, size: int, data_type: torch.dtype, pin_memory: bool
    ):
        """
        Return a tensor including `size` `device_type` elements on `device_type`.
        Delete the reference to the tenor in MemoryCache.
        If no cached tensor statisfied return None
        """
        if (device_type, data_type) not in self.cached_tensors:
            return None
        tensors = self.cached_tensors[(device_type, data_type)]
        i = -1
        for i in range(len(tensors)):
            if tensors[i].numel() == size:
                break
        if i == -1:
            return None
        new_tensor_ref = tensors[i]
        # delete the reference to tensors[i] in MemoryCache
        tensors.pop(i)
        return new_tensor_ref

    def recycle(self, payload) -> bool:
        """
        Recycle a payload tensor.
        If the cache is fulled, delete the payload.
        """
        device_type = payload.device
        data_type = payload.dtype
        if (device_type, data_type) not in self.cached_tensors and self.capacity_ > 0:
            self.cached_tensors[(device_type, data_type)] = [payload.zero_()]
            # print('first recycle payload')
            payload = None
        else:
            if len(self.cached_tensors[(device_type, data_type)]) == self.capacity_:
                # print("delete payload in MemoryCache")
                return False
            else:
                self.cached_tensors[(device_type, data_type)].append(payload.zero_())
                payload = None
        return True

    def delete(self, device_type):
        """
        Delete cached tensors on `device_type`
        """
        if (device_type, torch.float) in self.cached_tensors:
            del self.cached_tensors[(device_type, torch.float)]
        if (device_type, torch.float16) in self.cached_tensors:
            del self.cached_tensors[(device_type, torch.float)]
