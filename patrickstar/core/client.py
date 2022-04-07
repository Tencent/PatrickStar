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

import torch

from patrickstar.core.chunk_list import ChunkList
from patrickstar.core.const import ChunkState, TensorState
from patrickstar.core.parameter import register_param, is_registered, ParamType
from patrickstar.core.eviction_policy import LRUEvictionPolicy
from patrickstar.core.memtracer import RuntimeMemTracer
from patrickstar.utils import logger, get_world_size, get_rank, Metronome


class PatrickStarClient:
    r"""The client for managing chunks."""

    def __init__(self, local_rank, chunk_size, config={}):
        assert config is not None
        self.device = torch.device(f"cuda:{local_rank}")

        self.module = None

        self.metronome = Metronome()
        self.memtracer = RuntimeMemTracer(
            local_rank, self.metronome, config.get("memtracer", {})
        )
        self.eviction_policy = LRUEvictionPolicy(
            local_rank, self.metronome, self.memtracer
        )

        self.chunk_size = chunk_size
        self.chunk_list = ChunkList(
            self.chunk_size,
            self.memtracer,
            self.eviction_policy,
        )

    def is_warmup(self):
        return self.metronome.is_warmup

    def set_warmup(self, flag):
        self.metronome.is_warmup = flag

    def start_memtracer(self):
        self.memtracer.start(chunk_size=self.chunk_size)

    def new_dummy_chunk(self):
        r"""Append a dummy chunk to the corresponding chunk_list"""
        chunk = self.chunk_list.new_chunk(is_dummy=True)

        dummy = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float), requires_grad=False
        )
        # Add a dummy param to dummy chunk, so that the chunk can be set in HOLD state.
        register_param(dummy, ParamType.CHUNK_BASED, "dummy")
        chunk.add_param(dummy)
        logger.debug("Append a dummy chunk to the Chunk List")

    def append_params(self, params):
        r"""Append params to the last chunk.

        Append the whole list of param into the same chunk. If the last
        chunk doesn't fit, append a new chunk and try to insert params in it.
        """
        total_numel = 0
        for param in params:
            assert is_registered(param)
            total_numel += param.ps_attr.numel

        if total_numel > self.chunk_size:
            raise RuntimeError(
                f"Overall size of params is larger than the chunk size {self.chunk_size}."
            )

        if len(self.chunk_list) != 0:
            last_chunk = self.chunk_list[-1]
            if last_chunk.can_fit(total_numel):
                for param in params:
                    last_chunk.add_param(param)
                return

        chunk = self.chunk_list.new_chunk()
        for param in params:
            chunk.add_param(param)
        return

    def is_local_param(self, param):
        r"""Check if param is in local chunk"""
        chunk_id = param.ps_attr.info.chunk_id
        return self.chunk_list[chunk_id].is_local()

    def fetch_remote_chunks(self, comm_group, compute_device):
        r"""Fetch the remote chunks to local."""
        no_chunk_released = True
        for i in comm_group.elements:
            if self.chunk_list[i].get_state() == ChunkState.RELEASED:
                no_chunk_released = False
                break

        if no_chunk_released:
            return

        local_chunk_id = comm_group.elements[get_rank()]
        local_chunk = self.chunk_list[local_chunk_id]

        # Use collective communication to achieve the most efficient communication.
        # However, it is memory consumping. world_size chunks on GPU simutaneously.
        if self.is_warmup():
            self.eviction_policy.trace_access(local_chunk_id, compute_device)

        self.chunk_list.access_chunk(local_chunk, compute_device)
        local_chunk.pin()
        allgather_payload_buff = []
        for chunk_id in comm_group.elements:
            chunk = self.chunk_list[chunk_id]
            if chunk_id != local_chunk_id:
                self.chunk_list.try_allocate_payload(chunk, compute_device)
                chunk.pin()
                chunk.num_in_compute = 0

            allgather_payload_buff.append(chunk.payload)

        torch.distributed.all_gather(
            allgather_payload_buff,
            local_chunk.payload,
            async_op=False,
        )

        for chunk_id in comm_group.elements:
            self.chunk_list[chunk_id].unpin()

    def access_dist(self, param, compute_device):
        r"""Attach data to param.data, fetch from remote if chunk is released."""
        if param.ps_attr.is_torch_based():
            return

        chunk_id = param.ps_attr.info.chunk_id
        if get_world_size() > 1:
            self.fetch_remote_chunks(
                self.chunk_list[chunk_id].comm_info.group,
                compute_device,
            )

        self.access(param, compute_device)

    def access(self, param, compute_device):
        r"""Attach tensor to param.data."""
        if param.ps_attr.is_torch_based():
            return

        info = param.ps_attr.info
        numel = param.ps_attr.numel
        shape = param.ps_attr.shape
        start_offset = info.start_offset
        chunk_id = info.chunk_id
        if self.is_warmup():
            self.eviction_policy.trace_access(chunk_id, compute_device)

        chunk = self.chunk_list[chunk_id]
        self.chunk_list.access_chunk(chunk, compute_device)
        chunk.num_in_compute += 1
        param.data = chunk.payload.narrow(0, start_offset, numel).view(shape)
        param.ps_attr.state = TensorState.COMPUTE

    def get_grad(self, param):
        info = param.ps_attr.info
        numel = param.ps_attr.numel
        shape = param.ps_attr.shape
        start_offset = info.start_offset
        chunk_id = info.chunk_id

        grad_chunk = self.chunk_list.grad_chunks[chunk_id]
        # TODO(zilinzhu) Maybe move the grad chunks to GPU in the future.
        self.chunk_list.access_chunk(grad_chunk, torch.device("cpu:0"))
        return grad_chunk.payload.narrow(0, start_offset, numel).view(shape)

    def get_fp32(self, param):
        info = param.ps_attr.info
        numel = param.ps_attr.numel
        shape = param.ps_attr.shape
        start_offset = info.start_offset
        chunk_id = info.chunk_id

        fp32_chunk = self.chunk_list.fp32_chunks[chunk_id]
        # TODO(zilinzhu) Maybe move the fp32 chunks to GPU in the future.
        self.chunk_list.access_chunk(fp32_chunk, torch.device("cpu:0"))
        return fp32_chunk.payload.narrow(0, start_offset, numel).view(shape)

    def release(self, param):
        r"""Release the param in standalone environment.

        This means the param can be move to other device.
        """
        if param.ps_attr.is_torch_based():
            return

        # NOTE(jiaruifang) device must be the same as the origin param.
        # Or it will affect hook of param.grad_fn.next_functions[0][0].
        param.data = torch.tensor([], dtype=torch.half, device=param.device)

        chunk_id = param.ps_attr.info.chunk_id
        chunk = self.chunk_list[chunk_id]
        assert chunk.get_state() != ChunkState.RELEASED
        if param.ps_attr.state == TensorState.COMPUTE:
            chunk.num_in_compute -= 1
        param.ps_attr.state = TensorState.HOLD
