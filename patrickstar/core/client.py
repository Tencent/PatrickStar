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
from patrickstar.core.const import ChunkState, TensorState, TrainingStage
from patrickstar.core.hook import setup_patrickstar_hooks
from patrickstar.core.parameter import register_param, is_param_registered, ParamType
from patrickstar.core.eviction_policy import LatestAccessChunkEvictionPolicy
from patrickstar.core.memtracer import RuntimeMemTracer
from patrickstar.utils import logger, get_world_size, get_rank, log_dist, global_timer


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
            assert is_param_registered(param)
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

    def set_all_tensors_state_in_chunk(self, chunk_id, new_state):
        r"""Set the state of all tensors in a chunk.

        Notice that the state of the chunk will change as well.
        And this method has nothing to do with whether the payload of
        the chunk is allocated or not.
        """
        chunk = self.chunk_list[chunk_id]
        for param in chunk.params:
            old_state = param.ps_attr.get_state()
            chunk.update_state(old_state, new_state)
            param.ps_attr.set_state(new_state)

    def register_model_hook(self, model):
        setup_patrickstar_hooks(model, self)

    def is_local_param(self, param):
        r"""Check if param is in local chunk"""
        chunk_id = param.ps_attr.info.chunk_id
        return self.chunk_list[chunk_id].is_local()

    def _fetch_remote_chunks(
        self,
        chunk_id,
        chunk_id_list,
        local_chunk_id,
        compute_device,
        param_name,
    ):
        r"""Fetch the remote chunks to local.

        Args:
            chunk_id: the id of accessed chunk
            chunk_id_list: list of int. The id of the chunks in a same comm group.
            local_chunk_id: int. The id of the local chunk in the comm group.
            compute_device: :class:`torch.device`.
            param_name: str.
        """
        rank = get_rank()
        # During FWD, when there are param in the chunk group being visited for
        # the first time, collect the chunk group to local.
        # How can we determine if a chunk group is being visited for the first time,
        # so that we can trigger the correct allgather?
        # When the first param is visited, the remote chunk should be of state
        # RELEASED, therefore, we do the allgather when the state of chunks are
        # changing form HOLD_AFTER_FWD(HOLD_ADFTER_BWD) to RELEASED.
        no_chunk_released = True
        for i in chunk_id_list:
            if self.chunk_list[i].get_state() == ChunkState.RELEASED:
                no_chunk_released = False
                break
        if no_chunk_released:
            return

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_fetch_remote_chunks")

        logger.debug(
            f"rank {rank} fetch {param_name} remote chunks {chunk_id_list} local chunk {local_chunk_id}"
        )

        # Use collective communication to achieve the most efficient communication.
        # However, it is memory consumping. world_size chunks on GPU simutaneously.
        self.chunk_eviction_strategy.trace_access(local_chunk_id, compute_device)
        self.chunk_list.access_chunk(local_chunk_id, compute_device)
        self.chunk_list[local_chunk_id].pin()
        allgather_payload_buff = []
        comm_data_amount = 0
        for chunk_id in chunk_id_list:
            if chunk_id != local_chunk_id:
                self.chunk_list.try_best_allocate_payload(
                    self.chunk_list[chunk_id], compute_device
                )
                self.chunk_list[chunk_id].pin()
            self.set_all_tensors_state_in_chunk(chunk_id, TensorState.HOLD)
            allgather_payload_buff.append(self.chunk_list[chunk_id].payload)
        comm_data_amount = (
            len(allgather_payload_buff) * allgather_payload_buff[0].numel() * 2
        )  # half = 2 bytes
        for chunk_id in chunk_id_list:
            self.chunk_list[chunk_id].unpin()

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_fetch_remote_chunks_allgather")

        logger.debug(f"rank {rank} allgather {chunk_id_list}")
        torch.distributed.all_gather(
            allgather_payload_buff,
            self.chunk_list[local_chunk_id].payload,
            async_op=False,
        )

        allgather_payload_buff = []
        self.chunk_list[local_chunk_id].unpin()

        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_fetch_remote_chunks_allgather")
            global_timer.data_move_cnter.update(
                "CLIENT_fetch_remote_chunks_allgather", comm_data_amount
            )
        global_timer.my_timer.finish_profile("CLIENT_fetch_remote_chunks")

    def _access_tensor_in_chunk(self, param, compute_device, chunk_id):
        self.chunk_eviction_strategy.trace_access(chunk_id, compute_device)
        self.chunk_list.access_chunk(chunk_id, compute_device)
        # 2. Locate the param on the chunk.
        chunk = self.chunk_list[chunk_id]
        info = param.ps_attr.info
        start_offset = info.start_offset
        numel = param.ps_attr.numel

        param.ps_attr.set_tensor(chunk.payload.narrow(0, start_offset, numel))

        # 3. Change the state of param tensor.
        # The state of chunk should be determined by the state of its tensors.
        old_state = param.ps_attr.get_state()

        # If the old state was FREE, we need to fill the param to zero.
        if old_state == TensorState.FREE:
            param.ps_attr.access_tensor().zero_()

        # Change the state of param to COMPUTE.
        chunk.update_state(old_state, TensorState.COMPUTE)
        param.ps_attr.set_state(TensorState.COMPUTE)

        return param.ps_attr.access_tensor()

    def access_dist(
        self,
        param: torch.nn.Parameter,
        compute_device: torch.device,
    ) -> torch.Tensor:
        r"""Visit tensor of param in distributed environment.

        Notice that this method also works at standalone circumstances.

        Args:
            param: :class:`torch.nn.Parameter`. The param to visit.
            compute_device: :class:`torch.device`.
            training_stage: :class:`TrainingStage`.
        Returns:
            The tensor of the params.
        """

        assert is_param_registered(
            param
        ), "Client can only access_dist tensor registered for PatrickStar."
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return param.data

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_access_dist")

        # 1. Prepare the memory of the chunks. If the chunk is not one the
        #   compute device, then we need to move or allocate.
        chunk_id = param.ps_attr.info.chunk_id

        rank = get_rank()

        if get_world_size() > 1:
            chunk_id_list = self.chunk_list[chunk_id].comm_info.group.elements
            local_chunk_id = chunk_id_list[rank]

            logger.debug(
                f"rank {rank} access_dist access tensor {param.ps_attr.name} "
                f"local_chunk_id {local_chunk_id} chunk_id_list {chunk_id_list}"
            )

            # 1.2 Fetch the remote chunks to local.
            self._fetch_remote_chunks(
                chunk_id,
                chunk_id_list,
                local_chunk_id,
                compute_device,
                param.ps_attr.name,
            )
        else:
            local_chunk_id = chunk_id

        # collect the time a chunk has to be placed on compute-device
        # self.chunk_eviction_strategy.trace_access(local_chunk_id, compute_device)

        ret = self._access_tensor_in_chunk(param, compute_device, chunk_id)
        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_access_dist")
        return ret

    def access(
        self,
        param: torch.nn.Parameter,
        compute_device: torch.device,
    ):
        r"""Visit the data or grad of the local `param`.

        Steps:
            1. Find the chunk of the `param`.
            2.1 If the payload of the chunk exists,
                decide whether to move the chunk to `compute_device`.
            2.2 If the payload of the chunk does not exist,
                allocate the payload of on `compute_device`.
        Before moving or allocating, make sure there is enough space
        on the `compute_device`.

        Exceptions:
            The compute devices of 2 tensors in the same chunk is different.

        Notice that different from access_dist, this method will not do cross
        process communication.

        Args:
            param: :class:`torch.nn.Parameter`.
            compute_device: :class:`torch.device`.
        Returns:
            The tensor to access.
        """
        assert is_param_registered(
            param
        ), "client shall not access tensor not registered for PatrickStar"
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return param.data

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_access")

        # 1. Prepare the memory of the chunks. If the chunk is not one the
        #   compute device, then we need to move or allocate.
        chunk_id = param.ps_attr.info.chunk_id
        assert chunk_id is not None

        ret = self._access_tensor_in_chunk(param, compute_device, chunk_id)
        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_access")
        return ret

    def release_dist(
        self,
        param: torch.nn.Parameter,
        reset_to_state: TensorState,
        training_stage: TrainingStage,
    ):
        r"""Release the param in distributed environment.

        This means the data and grad of the param no longer need to
        stay in the current compute device.

        Steps:
            1. Update the state of tensor and chunk.
            2. If the chunk can be released,
                if `do_allreduce` is True, do reduce scatter to average the gradients.
                then released the payload.

        Args:
            param: :class:`torch.nn.Parameter`.
            reset_to_state: :class:`TensorState`. The state to reset tensor to.
            training_stage: :class:`TrainingStage`.
            do_allreduce: bool. Whether to do allreduce(reduce scatter).
                Notice that because user may use gradient checkpointing, the do_allreduce
                in TrainingStage.BWD doesn't equal to True.
            underutilze communication bandwidth.
        """
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_release_dist")
        rank = get_rank()

        assert isinstance(reset_to_state, TensorState)
        assert (
            training_stage == TrainingStage.FWD or training_stage == TrainingStage.BWD
        )
        assert torch.distributed.is_initialized()

        chunk_id = param.ps_attr.info.chunk_id
        chunk_id_list = self.chunk_list[chunk_id].comm_info.group.elements

        local_chunk_id = chunk_id_list[rank]

        logger.debug(
            f"rank {rank} release tensor {param.ps_attr.name} of chunk_id {chunk_id} to {reset_to_state}"
        )

        # Update the state of tensor and chunk.
        self.chunk_list[chunk_id].update_state(
            param.ps_attr.get_state(), reset_to_state
        )
        param.ps_attr.set_state(reset_to_state)

        # NOTE(jiaruifang) device must be the same as the origin param.
        # Or it will affect hook of param.grad_fn.next_functions[0][0].
        param.data = torch.tensor([], dtype=param.ps_attr.dtype, device=param.device)

        # Check if we finished using all tensors in all chunks of the chunk group,
        # then we can release the remote chunks.
        # The condition for releasing chunks are:
        #     FWD: All non-dummy chunks are of state HOLD_AFTER_FWD;
        #     BWD: All non-dummy chunks are of state HOLD_AFTER_BWD.
        world_size = get_world_size()
        if world_size > 1:
            if True:
                all_chunks_ready = True
                for i in chunk_id_list:
                    if training_stage == TrainingStage.FWD:
                        if (
                            not self.chunk_list[i].all_tensor_state(
                                TensorState.HOLD_AFTER_FWD
                            )
                            and not self.chunk_list[i].is_dummy()
                        ):
                            all_chunks_ready = False
                    elif training_stage == TrainingStage.BWD:
                        if (
                            not self.chunk_list[i].all_tensor_state(
                                TensorState.HOLD_AFTER_BWD
                            )
                            and not self.chunk_list[i].is_dummy()
                        ):
                            all_chunks_ready = False

                if all_chunks_ready:
                    # Remove the payload of remote chunks.
                    for i in chunk_id_list:
                        self.chunk_list[i].unpin()
                        if i != local_chunk_id:
                            logger.debug(f"rank {rank} remove payload of chunk_id {i}")
                            self.chunk_list[i].release_payload()
                            self.set_all_tensors_state_in_chunk(i, TensorState.FREE)

        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_release_dist")

    def release(
        self,
        param: torch.nn.Parameter,
        reset_to_state: TensorState = TensorState.HOLD,
    ):
        r"""Release the param in standalone environment.

        This means the data and grad of the param no longer need to
        stay in the current compute device.

        Steps:
            1. Update the state of tensor and chunk.
            2. If the chunk can be released released the payload.

        Args:
            param: :class:`torch.nn.Parameter`.
            reset_to_state: :class:`TensorState`. The state to reset tensor to.
        """
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return
        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_release")
        rank = self.local_rank
        assert isinstance(reset_to_state, TensorState)

        chunk_id = param.ps_attr.info.chunk_id
        logger.debug(
            f"rank {rank} release a tensor of chunk_id {chunk_id} to {reset_to_state}"
        )

        # Update the state of tensor and chunk.
        self.chunk_list[chunk_id].update_state(
            param.ps_attr.get_state(), reset_to_state
        )
        param.ps_attr.set_state(reset_to_state)

        # NOTE(jiaruifang) device must be the same as the origin param.
        # Or it will affect hook of param.grad_fn.next_functions[0][0].
        param.data = torch.tensor([], dtype=param.ps_attr.dtype, device=param.device)

        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_release")

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

    def display_chunk_info(self):
        logger.debug("Print chunk list info.")

        overall_size = 0
        overall_chunk_num = 0
        overall_utilization_ratio = 0.0
        max_utilization_ratio = 0
        for chunk in self.chunk_list.chunks:
            logger.debug(
                f"Chunk id {chunk.chunk_id}, state {chunk.get_state()}, "
                f"comm info {chunk.comm_info}, "
                f"capacity {chunk.capacity / 1024 / 1024} M elems, "
                f"device {chunk.get_device()}"
            )
            last_used_pos = chunk.end_pos
            logger.debug(
                f"chunk used {last_used_pos/1024/1024} M elem, "
                f"{last_used_pos/chunk.capacity * 100} %"
            )
            cur_util = last_used_pos / chunk.capacity
            max_utilization_ratio = max(cur_util, max_utilization_ratio)
            overall_utilization_ratio += cur_util
            overall_size += chunk.get_chunk_space()
            overall_chunk_num += 1

        log_dist(f"OVERALL CHUNK SIZE {overall_size / 1024 / 1024 / 1024} GB")
        log_dist(
            f"OVERALL UTILIZATION {overall_utilization_ratio / overall_chunk_num * 100} %"
        )
        log_dist(f"MAX UTILIZATION {max_utilization_ratio * 100} %")
