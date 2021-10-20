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

import time
import torch

from patrickstar.manager import PatrickStarManager
from patrickstar.profiler import profiler
from patrickstar.utils import logger, getsizeof
import patrickstar.utils.global_timer as global_timer
from .const import TensorStatus, ChunkStatus


class Chunk(object):
    def __init__(
        self,
        capacity: int,
        data_type: torch.dtype,
        chunk_id: int,
        local_rank: int = 0,
        is_dummy: bool = False,
    ):
        r"""
        Chunk is the minimal unit of the data transfer.
        It is a contiguous memory for saving tensors.
        To remove a tensor, we only need to set the status of the tensor to `FREE`.

        Chunk does no know if we are doing distributed training or not.
        Every process will observe its own chunk instances.

        Args:
            capacity: int. The maximum number of elements in the chunk.
            data_type: :class:`torch.dtype`.
            chunk_id: int.
            local_rank: int.
            is_dummy: bool.
        """
        self.chunk_id = chunk_id
        # payload numel does not equal to capacity. payload can be None.
        self.capacity = capacity
        self.data_type = data_type
        self.local_rank = local_rank
        self._is_dummy = is_dummy

        # the number of tensors of the chunk in each status
        self._status_dict = {
            TensorStatus.COMPUTE: 0,
            TensorStatus.HOLD: 0,
            TensorStatus.HOLD_AFTER_FWD: 0,
            TensorStatus.HOLD_AFTER_BWD: 0,
            TensorStatus.FREE: 0,
        }
        # the number of tensors that are not used in the forward calculation
        self.unused = 0

        self.payload = None
        self._time_profile = True

        self.gpu_access_moments = []
        self.cpu_access_moments = []
        self._pin_flag = False

    def append_moment(self, mom, compute_device):
        mgr = PatrickStarManager()
        assert mgr.is_warmup_training()

        access_moments = (
            self.gpu_access_moments
            if compute_device.type == "cuda"
            else self.cpu_access_moments
        )
        if len(access_moments) > 0 and mom == access_moments[-1]:
            return
        else:
            access_moments.append(mom)

    def next_accessed_mom(self, compute_device):
        r"""Get the next accessed moment after the warmup step.

        Args:
            compute_device: :class:`torch.device`.
        Returns:
            An int. The next access moment of the chunk. During warmup,
            return 0.
        """
        mgr = PatrickStarManager()
        access_moments = (
            self.gpu_access_moments
            if compute_device.type == "cuda"
            else self.cpu_access_moments
        )
        if mgr.is_nonwarmup_training():
            cur_mom = mgr.get_cur_mom()
            max_mom_small_than_cur = 0
            for i in access_moments:
                if i > cur_mom:
                    return i
                if i < cur_mom:
                    max_mom_small_than_cur = i
            return mgr.get_total_mom() + max_mom_small_than_cur
        else:
            return 0

    def display_access_mom_info(self):
        logger.info(f"\t {self.chunk_id} cpu_access_moments {self.cpu_access_moments}")
        logger.info(f"\t {self.chunk_id} gpu_access_moments {self.gpu_access_moments}")

    def is_dummy(self):
        return self._is_dummy

    def get_chunk_space(self):
        r"""Size of the chunk (Bytes)."""
        return getsizeof(self.data_type) * self.capacity

    def get_payload_space(self):
        r"""Size of the payload (Bytes)."""
        if self.payload is None:
            return 0
        else:
            return getsizeof(self.payload.dtype) * self.payload.numel()

    def pin(self):
        self._pin_flag = True

    def unpin(self):
        self._pin_flag = False

    def is_pin(self):
        return self._pin_flag

    def allocate_payload(self, device):
        r"""Allocate payload on device for the chunk.

        NOTE() This method does not check availability. Please check if
        there is enough room for the chunk.

        Args:
            device: :class:`torch.device`.
        """
        if self._time_profile:
            global_timer.my_timer.start_profile("CHUNK_allocate_payload")

        payload_size = self.capacity
        if device.type == "cpu":
            self.payload = torch.zeros(
                payload_size, dtype=self.data_type, device=device, pin_memory=True
            )
        else:
            self.payload = torch.zeros(
                payload_size, dtype=self.data_type, device=device
            )
        mgr = PatrickStarManager()
        mgr.add(device.type, self.get_payload_space())

        if profiler.started():
            profiler.chunk_life_cycle[self.chunk_id]["life_cycle"].append(
                (time.time(), "allocate", device)
            )

        if self._time_profile:
            global_timer.my_timer.finish_profile("CHUNK_allocate_payload")

    def release_payload(self):
        r"""Release the payload.

        NOTE() Please make sure all tensors are in the `FREE` status.
        """
        mgr = PatrickStarManager()
        mgr.delete(self.get_device().type, self.get_payload_space())

        # Remove the memory of the chunk.
        del self.payload
        self.payload = None

        if profiler.started():
            profiler.chunk_life_cycle[self.chunk_id]["life_cycle"].append(
                (time.time(), "release", None)
            )

    def update_status(self, old_status, new_status):
        r"""Update the status counter of tensors of the chunk.

        Args:
            old_status: :class:`TensorStatus`.
            new_status: :class:`TensorStatus`.
        """
        self._status_dict[old_status] -= 1
        self._status_dict[new_status] += 1

    def get_status(self):
        """
        When payload is None, the status is `RELEASED`,
        otherwise, status of the chunk is decided by its tensors.

        Returns:
            :class:`ChunkStatus`.
        """
        if self.payload is None:
            return ChunkStatus.RELEASED

        # Distributed training need to fix the chunk on the compute device.
        if self._status_dict[TensorStatus.COMPUTE] > 0:
            return ChunkStatus.COMPUTE
        elif self._status_dict[TensorStatus.HOLD] > 0:
            return ChunkStatus.HOLD
        elif self._status_dict[TensorStatus.HOLD_AFTER_FWD] > 0:
            return ChunkStatus.HOLD_AFTER_FWD
        elif self._status_dict[TensorStatus.HOLD_AFTER_BWD] > 0:
            return ChunkStatus.HOLD_AFTER_BWD
        else:
            return ChunkStatus.FREE

    def all_tensor_status(self, status):
        r"""If all tensors are in the status or `FREE`.

        Args:
            status: :class:`TensorStatus`.
        Return:
            bool.
        """
        for k, v in self._status_dict.items():
            if k != TensorStatus.FREE and k != status:
                if v != 0:
                    # Ignore the unused tensors.
                    if k == TensorStatus.HOLD and v == self.unused:
                        continue
                    return False
        return True

    def set_unused(self):
        r"""
        After forward calculation, the tensors in `HOLD` status are the ones
        that are not used. Remember them for the release.
        NOTE() This function can only be called at the end of forward calculation.
        """
        # TODO(zilinzhu) Find a better way to represent the unused tensors
        self.unused = self._status_dict[TensorStatus.HOLD]

    def move(self, target_device: torch.device):
        r"""
        Move the chunk to `target_device`.
        NOTE() Please check if the `target_device` has enough room before.

        Args:
            target_device: :class:`torch.device`.
        """
        if self.get_device() is None:
            logger.warning(f"chunk move payload None to {target_device}")
            return
        if self.get_device() == target_device:
            return
        if self._time_profile:
            if target_device.type == "cuda":
                global_timer.my_timer.start_profile("chunk_cpu_gpu_move")
            else:
                global_timer.my_timer.start_profile("chunk_gpu_cpu_move")
        src_device = self.get_device()
        mgr = PatrickStarManager()

        logger.debug(
            f"move chunk {self.chunk_id}, which has {self.payload.numel() / 1e6} M {self.payload.dtype} elements, "
            f"from {src_device} to {target_device}, "
            f"used mem {mgr.used_chunk_mem(target_device.type) / 1e6} MB"
        )

        # TODO(jiaruifang) asyc copy.
        if target_device.type == "cpu":
            pinned_payload_cpu = torch.empty(
                self.payload.shape,
                dtype=self.payload.dtype,
                device="cpu:0",
                pin_memory=True,
            )
            with torch.cuda.stream(mgr.copy_stream):
                pinned_payload_cpu.copy_(self.payload)
            self.payload = pinned_payload_cpu
        elif target_device.type == "cuda":
            self.payload = self.payload.pin_memory()
            with torch.cuda.stream(mgr.copy_stream):
                self.payload = self.payload.to(target_device)

        mgr.delete(src_device.type, self.get_payload_space())
        mgr.add(target_device.type, self.get_payload_space())

        if self._time_profile:
            if target_device.type == "cuda":
                global_timer.my_timer.finish_profile("chunk_cpu_gpu_move")
                global_timer.data_move_cnter.update(
                    "chunk_cpu_gpu_move", self.get_payload_space()
                )
            elif target_device.type == "cpu":
                global_timer.my_timer.finish_profile("chunk_gpu_cpu_move")
                global_timer.data_move_cnter.update(
                    "chunk_gpu_cpu_move", self.get_payload_space()
                )

        if profiler.started():
            if len(profiler.chunk_life_cycle[self.chunk_id]["life_cycle"]) == 0:
                raise RuntimeError(
                    f"Chunk {self.chunk_id} allocation time is not recorded. "
                    f"You may need to put profiler.start() before initialize_engine "
                )
            profiler.chunk_life_cycle[self.chunk_id]["life_cycle"].append(
                (time.time(), "move", target_device)
            )

    def get_device(self):
        r"""Get device of the payload of chunk, return None if not allocated."""
        if self.payload is not None:
            return self.payload.device
        else:
            return None
