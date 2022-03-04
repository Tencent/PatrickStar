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

from typing import List

import torch

from patrickstar.core.chunk_list import ChunkList
from patrickstar.core.const import ChunkState, TensorState
from patrickstar.core.hook import setup_patrickstar_hooks
from patrickstar.core.parameter import register_param, is_registered, ParamType
from patrickstar.core.eviction_policy import LatestAccessChunkEvictionPolicy
from patrickstar.core.memtracer import RuntimeMemTracer
from patrickstar.utils import logger, get_world_size, get_rank, global_timer


class PatrickStarClient(object):
    r"""The client for managing chunks."""

    def __init__(self, rank: int, chunk_size: int, config=None):
        self.local_rank = rank
        self.device = torch.device(f"cuda:{rank}")

        self.module = None

        default_tracer_config = {
            "use_async_mem_monitor": True,
            "warmup_gpu_chunk_mem_ratio": 0.1,
            "overall_gpu_mem_ratio": 0.8,
            "overall_cpu_mem_ratio": 0.8,
            "margin_use_ratio": 0.8,
            "with_static_partition": False,
        }
        if config is not None:
            tracer_config = config.get("mem_tracer", None)
            for k, v in default_tracer_config.items():
                if k not in tracer_config:
                    tracer_config[k] = v
        else:
            tracer_config = default_tracer_config

        self.mem_tracer = RuntimeMemTracer(self.local_rank, tracer_config)

        self.chunk_eviction_strategy = LatestAccessChunkEvictionPolicy(
            self.mem_tracer.metronome
        )

        self.chunk_size = chunk_size
        self.chunk_list = ChunkList(
            self.local_rank,
            self.mem_tracer,
            self.chunk_eviction_strategy,
            self.chunk_size,
        )
        self._time_profile = True

        self.dummy_param_list = []

        # for post backward hook
        self.grad_accs = []

    # expose APIs from metrome ti client
    def training_stage(self):
        return self.mem_tracer.metronome.training_stage()

    def set_training_phase(self, phase):
        self.mem_tracer.metronome.set_training_phase(phase)

    def set_warmup(self, flag):
        self.mem_tracer.metronome.set_warmup(flag)

    def is_warmup(self):
        return self.mem_tracer.is_warmup()

    def trigger_memory_tracing(self):
        self.mem_tracer.trace_memory()

    def start_mem_tracer(self):
        """
        Memory tracer start to work!
        """
        self.mem_tracer.start_train(
            param_fp16_chunk_size=self.param_fp16_chunks_max_mem_usage(),
            chunk_size=self.chunk_size,
        )

    def new_dummy_chunk(self):
        r"""Append a dummy chunk to the corresponding chunk_list"""
        chunk = self.chunk_list.new_chunk(is_dummy=True)

        dummy = torch.nn.Parameter(
            torch.tensor([], dtype=torch.float), requires_grad=False
        )
        # Add a dummy param to dummy chunk, so that the chunk can be set in HOLD state.
        register_param(dummy, ParamType.CHUNK_BASED, "dummy")
        self.dummy_param_list.append(dummy)
        chunk.add_param(dummy)
        logger.debug("Append a dummy chunk to the Chunk List")

    def append_params(
        self,
        params: List[torch.nn.Parameter],
    ):
        r"""Append params to the last chunk.

        Append the whole list of param into the same chunk. If the last
        chunk doesn't fit, append a new chunk and try to insert params in it.

        Args:
            param_list: list of `torch.nn.Parameter`.
        """
        total_numel = 0
        for param in params:
            assert is_registered(param)
            total_numel += param.ps_attr.numel

        if len(self.chunk_list) != 0:
            last_chunk = self.chunk_list[-1]
            if last_chunk.can_fit(total_numel):
                for param in params:
                    last_chunk.add_param(param)
                return

        chunk = self.chunk_list.new_chunk()
        if not chunk.can_fit(total_numel):
            raise RuntimeError(
                f"Overall size of params is larger than the chunk size {chunk.capacity}."
            )
        for param in params:
            chunk.add_param(param)
        return

    def param_fp16_chunks_max_mem_usage(self):
        r"""Return the total memory used by param fp16 chunks in bytes.

        In distributed environment, the return value includes remote chunks
        from allgather.
        """
        world_size = get_world_size()
        # non MSC has to cache work_size - 1 buffer.
        return (
            len(self.chunk_list.chunks) * self.chunk_size * 2 / world_size
            + (world_size - 1) * self.chunk_size * 2
        )

    def register_model_hook(self, model):
        setup_patrickstar_hooks(model, self)

    def is_local_param(self, param):
        r"""Check if param is in local chunk"""
        chunk_id = param.ps_attr.info.chunk_id
        return self.chunk_list[chunk_id].is_local()

    def fetch_remote_chunks(
        self,
        comm_group,
        compute_device,
    ):
        r"""Fetch the remote chunks to local."""
        no_chunk_released = True
        for i in comm_group.elements:
            if self.chunk_list[i].get_state() == ChunkState.RELEASED:
                no_chunk_released = False
                break

        if no_chunk_released:
            return

        local_chunk_id = comm_group.elements[get_rank()]

        if self._time_profile:
            global_timer.start_profile("CLIENT_fetch_remote_chunks")

        # Use collective communication to achieve the most efficient communication.
        # However, it is memory consumping. world_size chunks on GPU simutaneously.
        self.chunk_eviction_strategy.trace_access(local_chunk_id, compute_device)
        self.chunk_list.access_chunk(local_chunk_id, compute_device)
        self.chunk_list[local_chunk_id].pin()
        allgather_payload_buff = []
        for chunk_id in comm_group.elements:
            chunk = self.chunk_list[chunk_id]
            if chunk_id != local_chunk_id:
                self.chunk_list.try_allocate_payload(chunk, compute_device)
                chunk.pin()
                chunk.num_in_compute = 0
                for param in chunk.params:
                    param.ps_attr.state = TensorState.HOLD

            allgather_payload_buff.append(chunk.payload)

        if self._time_profile:
            global_timer.start_profile("CLIENT_fetch_remote_chunks_allgather")

        torch.distributed.all_gather(
            allgather_payload_buff,
            self.chunk_list[local_chunk_id].payload,
            async_op=False,
        )

        for chunk_id in comm_group.elements:
            self.chunk_list[chunk_id].unpin()

        if self._time_profile:
            global_timer.finish_profile("CLIENT_fetch_remote_chunks_allgather")
        global_timer.finish_profile("CLIENT_fetch_remote_chunks")

    def access_dist(self, param, compute_device):
        r"""Attach data to param.data, fetch from remote if chunk is released."""
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
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
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return

        chunk_id = param.ps_attr.info.chunk_id
        self.chunk_eviction_strategy.trace_access(chunk_id, compute_device)
        self.chunk_list.access_chunk(chunk_id, compute_device)
        # 2. Locate the param on the chunk.
        chunk = self.chunk_list[chunk_id]
        info = param.ps_attr.info
        numel = param.ps_attr.numel
        shape = param.ps_attr.shape
        start_offset = info.start_offset

        param.data = chunk.payload.narrow(0, start_offset, numel).view(shape)

        # Change the state of param to COMPUTE.
        chunk.num_in_compute += 1
        param.ps_attr.state = TensorState.COMPUTE

    def release(self, param):
        r"""Release the param in standalone environment.

        This means the param can be move to other device.
        """
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return

        chunk_id = param.ps_attr.info.chunk_id
        chunk = self.chunk_list[chunk_id]
        assert chunk.get_state() != TensorState.RELEASED
        if param.ps_attr.state == TensorState.COMPUTE:
            chunk.num_in_compute -= 1
        param.ps_attr.state = TensorState.HOLD

        # NOTE(jiaruifang) device must be the same as the origin param.
        # Or it will affect hook of param.grad_fn.next_functions[0][0].
        param.data = torch.tensor([], dtype=param.ps_attr.dtype, device=param.device)

    def get_overall_chunk_size(self):
        """
        return the overall size of all chunks and
        the overall chunk utilization excluding fragments.
        Excepting the dummy chunk if using MSC.
        """
        overall_size = 0
        overall_chunk_num = 0
        overall_utilization_ratio = 0.0
        for chunk in self.chunk_list.chunks:
            last_used_pos = chunk.end_pos
            overall_utilization_ratio += last_used_pos / chunk.capacity
            overall_size += chunk.get_chunk_space()
            overall_chunk_num += 1
        overall_utilization_ratio /= overall_chunk_num
        return overall_size, overall_utilization_ratio
