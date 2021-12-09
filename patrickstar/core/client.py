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

import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import logger, get_world_size, get_rank
from .chunk_list import ChunkList, ChunkType
from .chunk_tensor_index import ChunkTensorIndex
from .const import AccessType, ChunkState, TensorState, TrainingStage
from .hook import setup_patrickstar_hooks
from .parameter import register_param, is_param_registered, ParamType
from .eviction_policy import LatestAccessChunkEvictionPolicy
from patrickstar.core.memtracer import RuntimeMemTracer


class PatrickStarClient(object):
    r"""The client for managing chunks."""

    def __init__(self, rank: int, default_chunk_size: int, config=None):
        self.local_rank = rank
        self.device = torch.device(f"cuda:{rank}")

        self.module = None

        default_tracer_config = {
            "use_async_mem_monitor": True,
            "warmup_gpu_chunk_mem_ratio": 0.1,
            "overall_gpu_mem_ratio": 0.8,
            "overall_cpu_mem_ratio": 0.8,
            "margin_use_ratio": 0.8,
            "use_fake_dist": False,
            "with_static_partition": False,
        }
        default_opt_config = {
            "with_mem_saving_comm": False,
            "with_mem_cache": False,
            "with_async_move": False,
        }
        if config is not None:
            tracer_config = config.get("mem_tracer", None)
            for k, v in default_tracer_config.items():
                if k not in tracer_config:
                    tracer_config[k] = v
            opt_config = config.get("opts", None)
        else:
            tracer_config = default_tracer_config
            opt_config = default_opt_config

        self.mem_tracer = RuntimeMemTracer(self.local_rank, tracer_config)
        self.opt_config = opt_config

        self.chunk_eviction_strategy = LatestAccessChunkEvictionPolicy(
            self.mem_tracer.metronome
        )

        self.default_chunk_size = default_chunk_size
        self.chunk_tensor_index = ChunkTensorIndex(self.default_chunk_size)
        self.chunk_list = ChunkList(
            self.local_rank,
            self.mem_tracer,
            self.chunk_eviction_strategy,
            self.opt_config["with_mem_cache"],
            self.opt_config["with_async_move"],
        )
        if self.opt_config["with_mem_cache"]:
            print("[CONFIG] USING MEM CACHE")
        self._time_profile = True

        if torch.distributed.is_initialized():
            self.cpu_comm_group = torch.distributed.new_group(backend="gloo")
        else:
            self.cpu_comm_group = None

        self.dummy_param_list = []
        # The list of torch params that will register allreduce hook
        self.torch_param_allreduce_list = []
        self.param_fp16_to_param_fp32_map = {}
        self.chunk_based_param_fp16 = []

        # for post backward hook
        self.grad_accs = []

        # A set to record chunks that are being visited.
        self.visiting_chunk = {}

    def visiting_finish(self, chunk_id):
        r"""
        Used for memory saving comm.
        Finish visiting of chunk_id.
        The remote chunk is released.
        """
        assert chunk_id in self.visiting_chunk
        self.visiting_chunk.pop(chunk_id)

    def visiting_start(self, chunk_id):
        r"""
        Start visiting of chunk_id.
        """
        self.visiting_chunk[chunk_id] = 1

    def is_visiting(self, chunk_id):
        return chunk_id in self.visiting_chunk

    def reset_visited_chunk(self):
        self.visiting_chunk = {}

    # expose APIs from metrome ti client
    def training_stage(self):
        return self.mem_tracer.metronome.training_stage()

    def set_training_phase(self, phase):
        self.mem_tracer.metronome.set_training_phase(phase)

    def set_warmup(self, flag):
        self.mem_tracer.metronome.set_warmup(flag)

    def is_warmup(self):
        return self.mem_tracer.is_warmup()

    def init(self, model, optimizer):
        r"""Initialize and store model and optimizer"""

        self.module = model
        self.optimizer = optimizer
        if get_rank() == 0:
            self.display_chunk_info()
        # Here we register the forward and backward hooks.
        self.register_model_hook(model)

    def trigger_memory_tracing(self):
        self.mem_tracer.trace_memory()

    def adjust_chunk_layout(self):
        """ "
        Adjust chunk layout in heterogenous memory space
        according to the runtime memory statictis.
        """
        if self.mem_tracer.metronome.is_warmup():
            return
        gpu_device = torch.device(f"cuda:{self.local_rank}")
        next_mom = self.mem_tracer.metronome.next_moment()
        # cur_mom = self.mem_tracer.metronome.moment()
        gpu_next_mom_ava_chunk_mem = (
            self.mem_tracer._overall_gpu_mem
            - self.mem_tracer.gpu_sys_used_list[next_mom]
        )
        gpu_cur_mom_used_chunk_mem = self.chunk_list.get_chunk_memory_used(gpu_device)
        if gpu_next_mom_ava_chunk_mem < gpu_cur_mom_used_chunk_mem:
            offload_size = gpu_cur_mom_used_chunk_mem - gpu_next_mom_ava_chunk_mem
            # NOTE() Here will lead to GPU <-> CPU memory movement.
            self.chunk_list.make_room(offload_size, gpu_device)

    def start_mem_tracer(self):
        """
        Memory tracer start to work!
        """
        self.mem_tracer.start_train(
            param_fp16_chunk_size=self.param_fp16_chunks_max_mem_usage(),
            chunk_size=self.default_chunk_size,
        )

    def append_chunk(self, data_type, chunk_type, is_dummy=False):
        r"""Append a new chunk to chunk_list and chunk_tensor_index.

        Args:
            data_type: :class:`torch.dtype`.
            chunk_type: :class:`ChunkType`.
            is_dummy: bool.
        Returns:
            chunk_id of the newly created chunk and comm_info.
        """
        chunk_id = self.chunk_list.generate_chunk_id()
        comm_info = self.chunk_list.new_chunk(
            chunk_id,
            self.default_chunk_size,
            data_type,
            is_dummy=is_dummy,
            chunk_type=chunk_type,
        )
        self.chunk_tensor_index.add_chunk(chunk_id, comm_info)
        return chunk_id, comm_info

    def append_dummy_chunk(self, data_type: torch.dtype, chunk_type: ChunkType):
        r"""Append a dummy chunk to the corresponding chunk_list"""
        chunk_id, comm_info = self.append_chunk(torch.half, chunk_type, is_dummy=True)

        dummy = torch.nn.Parameter(
            torch.tensor([], dtype=data_type), requires_grad=False
        )
        # Add a dummy param to dummy chunk, so that the chunk can be set in HOLD state.
        register_param(
            dummy, ParamType.CHUNK_BASED, torch.half, f"dummy_{comm_info.group_id}"
        )
        self.dummy_param_list.append(dummy)
        self.chunk_tensor_index.add_tensor(
            chunk_id,
            self.dummy_param_list[-1].ps_attr.data_id(),
            0,
            dummy.numel(),
            self.dummy_param_list[-1],
            AccessType.DATA,
        )

        logger.info(
            f"Append a dummy chunk to the Chunk List {chunk_type} "
            f"comm info {comm_info}"
        )

    def delete_param(self, param, access_type):
        """
        TODO(jiaruifang) Remove tensor of the param
        """
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        self.chunk_tensor_index.delete_tensor(chunk_id, param, access_type)

    def append_tensor(
        self,
        param_list: List[torch.nn.Parameter],
        data_type: torch.dtype,
        access_type: AccessType,
        chunk_type: ChunkType,
    ):
        r"""Append params to the last chunk of type `chunk_type`.

        Append the whole list of param into the same chunk. If the last
        chunk doesn't fit, append a new chunk and try to insert params in it.

        Args:
            param_list: list of `torch.nn.Parameter`.
            data_type: :class:`torch.dtype`. Can be different from param.
            access_type: :class:`AccessType`.
            chunk_type: :class:`ChunkType`.
        """
        assert isinstance(data_type, torch.dtype)
        assert isinstance(access_type, AccessType)
        if not self.chunk_list.is_empty(chunk_type):
            last_chunk_id = self.chunk_list.last_chunk_id(chunk_type)
            if self.chunk_tensor_index.try_insert_tensor_list(
                last_chunk_id, param_list, access_type
            ):
                return
        chunk_id, _ = self.append_chunk(data_type, chunk_type)
        if not self.chunk_tensor_index.try_insert_tensor_list(
            chunk_id, param_list, access_type
        ):
            raise RuntimeError(
                f"Can not append a tensor to chunk_tensor_index. "
                f"Overall size of param list is larger than the default chunk size {self.default_chunk_size}."
            )
        return

    def append_tensor_as_ref(
        self, param, data_type, access_type, chunk_type, ref_param
    ):
        r"""Append param to the last chunk with regard to the ref_param's location.

        When adding optimizer params, e.g. the variance and momentum of adam to
        chunk list, we hope they are of the same order as their corresponding
        fp16 params. Here the `param` is the optimizer params and `ref_param` is
        the fp16 param.
        Notice that the chunk_id of param and ref_param are different.

        Args:
            param: :class:`torch.nn.Parameter`.
            data_type: :class:`torch.dtype`. Can be different from param.
            access_type: :class:`AccessType`.
            chunk_type: :class:`ChunkType`.
            ref_param: :class:`torch.nn.Parameter`.
        """
        chunk_id = self.chunk_tensor_index.get_optimizer_state_chunk_id(
            ref_param, access_type, chunk_type
        )
        if chunk_id is None:
            chunk_id, _ = self.append_chunk(data_type, chunk_type)
        if not self.chunk_tensor_index.try_insert_tensor(chunk_id, param, access_type):
            raise RuntimeError("Failed to insert optimizer param w.r.t its ref_param.")
        self.chunk_tensor_index.register_optimizer_state_chunk_id(
            ref_param, access_type, chunk_type, chunk_id
        )

    def param_fp16_chunks_max_mem_usage(self):
        r"""Return the total memory used by param fp16 chunks in bytes.

        In distributed environment, the return value includes remote chunks
        from allgather.
        """
        world_size = get_world_size()
        # 本进程自己管理的Chunk，和Group Chunk Buff会分配的Chunk
        return (
            self.chunk_tensor_index.chunk_num(ChunkType.PARAM_FP16)
            * self.default_chunk_size
            * 2
            / world_size
            + (world_size - 1) * self.default_chunk_size * 2
        )

    def set_all_tensors_state_in_chunk(self, chunk_id, new_state):
        r"""Set the state of all tensors in a chunk.

        Notice that the state of the chunk will change as well.
        And this method has nothing to do with whether the payload of
        the chunk is allocated or not.
        """
        for info in self.chunk_tensor_index.generate_tensor_info_in_order(chunk_id):
            param = info.param
            access_type = info.access_type
            old_state = param.ps_attr.get_state(access_type)
            self.chunk_list.update_state(chunk_id, old_state, new_state)
            param.ps_attr.set_state(new_state, access_type)

    def register_model_hook(self, model):
        setup_patrickstar_hooks(model, self)

    def chunk_ids_generator(self, chunk_type: ChunkType):
        return self.chunk_list.chunk_ids_generator(chunk_type)

    def is_local_param(self, param, access_type):
        r"""Check if param is in local chunk"""
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        return self.chunk_tensor_index.is_local_chunk(chunk_id)

    def _fetch_remote_chunks(
        self,
        chunk_id,
        chunk_id_list,
        local_chunk_id,
        compute_device,
        with_mem_saving_comm: bool,
        param_name,
        training_stage,
    ):
        r"""Fetch the remote chunks to local.

        Args:
            chunk_id: the id of accessed chunk
            chunk_id_list: list of int. The id of the chunks in a same comm group.
            local_chunk_id: int. The id of the local chunk in the comm group.
            compute_device: :class:`torch.device`.
            with_mem_saving_comm: using the memory saving communication pattern or not.
            param_name: str.
        """
        rank = get_rank()
        if with_mem_saving_comm:
            # Use memory saving communication pattern.
            # Bcast chunk from the src gpu to the others.

            # check the chunk_id is the first to be visited.
            # local chunk as HOLD, remote chunk as RELEASED
            if self.is_visiting(chunk_id):
                return

            self.visiting_start(chunk_id)
            # Find the source rank to bcast its local chunk, which owned by the gpu.
            src_rank = -1
            for cur_rank, cur_chunk_id in enumerate(chunk_id_list):
                if chunk_id == cur_chunk_id:
                    src_rank = cur_rank

            # If the gpu owns the chunk (local rank), access it.
            # If the gpu do not own the chunk (remote chunk), allocate memory.
            if src_rank == rank:
                self.chunk_list.access_chunk(chunk_id, compute_device)
            else:
                self.chunk_list.try_best_allocate_payload(
                    self.chunk_list[chunk_id], compute_device
                )
            if self._time_profile:
                global_timer.my_timer.start_profile(
                    "CLIENT_fetch_remote_chunks_broadcast"
                )

            # Do Bcast from gpu owns chunk to gpu do not own it.
            torch.distributed.broadcast(
                self.chunk_list[chunk_id].payload,
                src=src_rank,
                async_op=False,
            )
            if self._time_profile:
                global_timer.data_move_cnter.update(
                    "CLIENT_fetch_remote_chunks_broadcast",
                    self.chunk_list[chunk_id].payload.numel() * 2,
                )
                global_timer.my_timer.finish_profile(
                    "CLIENT_fetch_remote_chunks_broadcast"
                )
            # set the chunk as HOLD, therefore it can be offloaded to CPU.
            self.set_all_tensors_state_in_chunk(chunk_id, TensorState.HOLD)
        else:
            # During FWD, when there are param in the chunk group being visited for
            # the first time, collect the chunk group to local.
            # How can we determine if a chunk group is being visited for the first time,
            # so that we can trigger the correct allgather?
            # When the first param is visited, the remote chunk should be of state
            # RELEASED, therefore, we do the allgather when the state of chunks are
            # changing form HOLD_AFTER_FWD(HOLD_ADFTER_BWD) to RELEASED.
            has_released_chunk = False
            for i in chunk_id_list:
                if self.chunk_list[i].get_state() == ChunkState.RELEASED:
                    has_released_chunk = True
                    break
            if not has_released_chunk:
                return

            if self._time_profile:
                global_timer.my_timer.start_profile("CLIENT_fetch_remote_chunks")

            logger.debug(
                f"rank {rank} fetch {param_name} remote chunks {chunk_id_list} local chunk {local_chunk_id}"
            )

            # Use collective communication to achieve the most efficient communication.
            # However, it is memory consumping. world_size chunks on GPU simutaneously.
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

            assert (
                torch.distributed.is_initialized()
            ), "torch distributed is not initialized during allgather"
            if self._time_profile:
                global_timer.my_timer.start_profile(
                    "CLIENT_fetch_remote_chunks_allgather"
                )

            logger.info(f"rank {rank} allgather {chunk_id_list}")
            torch.distributed.all_gather(
                allgather_payload_buff,
                self.chunk_list[local_chunk_id].payload,
                async_op=False,
            )

            allgather_payload_buff = []
            self.chunk_list[local_chunk_id].unpin()

            if self._time_profile:
                global_timer.my_timer.finish_profile(
                    "CLIENT_fetch_remote_chunks_allgather"
                )
                global_timer.data_move_cnter.update(
                    "CLIENT_fetch_remote_chunks_allgather", comm_data_amount
                )
            global_timer.my_timer.finish_profile("CLIENT_fetch_remote_chunks")

    def _access_tensor_in_chunk(self, param, access_type, compute_device, chunk_id):
        self.chunk_list.access_chunk(chunk_id, compute_device)
        # 2. Locate the param on the chunk.
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.chunk_tensor_index.get_tensor_info(tensor_id)
        start_offset = info.start_offset
        numel = info.numel
        assert numel == param.ps_attr.numel, f"{numel} vs {param.ps_attr.numel}"

        param.ps_attr.set_tensor(
            self.chunk_list[chunk_id].payload.narrow(0, start_offset, numel),
            access_type,
        )

        # 3. Change the state of param tensor.
        # The state of chunk should be determined by the state of its tensors.
        old_state = param.ps_attr.get_state(access_type)

        # If the old state was FREE, we need to fill the param to zero.
        if old_state == TensorState.FREE:
            param.ps_attr.access_tensor(access_type).zero_()

        # Change the state of param to COMPUTE.
        self.chunk_list.update_state(chunk_id, old_state, TensorState.COMPUTE)
        param.ps_attr.set_state(TensorState.COMPUTE, access_type)

        return param.ps_attr.access_tensor(access_type)

    def access_dist(
        self,
        param: torch.nn.Parameter,
        access_type: AccessType,
        compute_device: torch.device,
        with_mem_saving_comm: bool,
        training_stage: TrainingStage,
    ) -> torch.Tensor:
        r"""Visit tensor of param in distributed environment.

        Notice that this method also works at standalone circumstances.

        Args:
            param: :class:`torch.nn.Parameter`. The param to visit.
            access_type: :class:`AccessType`.
            compute_device: :class:`torch.device`.
            training_stage: :class:`TrainingStage`.
        Returns:
            The tensor of the params.
        """

        assert is_param_registered(
            param
        ), "Client can only access_dist tensor registered for PatrickStar."
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            if access_type == AccessType.DATA:
                return param.data
            elif access_type == AccessType.GRAD:
                return param.grad
            else:
                raise RuntimeError(f"{access_type} is not supported")

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_access_dist")

        # 1. Prepare the memory of the chunks. If the chunk is not one the
        #   compute device, then we need to move or allocate.
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        rank = get_rank()

        if get_world_size() > 1:
            chunk_id_list = self.chunk_tensor_index.chunk_ids_of_comm_group(chunk_id)
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
                with_mem_saving_comm,
                param.ps_attr.name,
                training_stage,
            )
        else:
            local_chunk_id = chunk_id

        # collect the time a chunk has to be placed on compute-device
        self.chunk_eviction_strategy.trace_access(local_chunk_id, compute_device)

        ret = self._access_tensor_in_chunk(param, access_type, compute_device, chunk_id)
        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_access_dist")
        return ret

    def access(
        self,
        param: torch.nn.Parameter,
        access_type: AccessType,
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
            access_type: :class:`AccessType`.
            compute_device: :class:`torch.device`.
        Returns:
            The tensor to access.
        """
        assert is_param_registered(
            param
        ), "client shall not access tensor not registered for PatrickStar"
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            if access_type == AccessType.DATA:
                return param.data
            elif access_type == AccessType.GRAD:
                return param.grad
            else:
                raise RuntimeError(f"{access_type} is not supported")

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_access")

        # 1. Prepare the memory of the chunks. If the chunk is not one the
        #   compute device, then we need to move or allocate.
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        # collect the time a chunk has to be placed on compute-device
        self.chunk_eviction_strategy.trace_access(chunk_id, compute_device)

        if chunk_id is None:
            raise RuntimeError(
                "FP16 training shall not meet tensors with no chunk assigned. "
                "Every tensor has to be assigned to a chunk during a tensor-chunk-mapping "
                "process before training."
            )

        ret = self._access_tensor_in_chunk(param, access_type, compute_device, chunk_id)
        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_access")
        return ret

    def access_data(self, param: torch.nn.Parameter, compute_device: torch.device):
        r"""move the PSTensor of param.data to `compute_device`."""
        return self.access(param, AccessType.DATA, compute_device)

    def access_grad(self, param: torch.nn.Parameter, compute_device: torch.device):
        r"""move the PSTensor of param.data to `compute_device`.

        NOTE() The device of grad should be determined by the device of param.data.
        """
        return self.access(param, AccessType.GRAD, compute_device)

    def release_dist(
        self,
        param: torch.nn.Parameter,
        access_type: AccessType,
        reset_to_state: TensorState,
        training_stage: TrainingStage,
        do_allreduce: bool,
        with_mem_saving_comm: bool = False,
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
            access_type: :class:`AccessType`.
            reset_to_state: :class:`TensorState`. The state to reset tensor to.
            training_stage: :class:`TrainingStage`.
            do_allreduce: bool. Whether to do allreduce(reduce scatter).
                Notice that because user may use gradient checkpointing, the do_allreduce
                in TrainingStage.BWD doesn't equal to True.
            with_mem_saving_comm: book. Use memory saving communication pattern. Save memory but
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

        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        chunk_id_list = self.chunk_tensor_index.chunk_ids_of_comm_group(chunk_id)

        local_chunk_id = chunk_id_list[rank]

        logger.debug(
            f"rank {rank} release tensor {param.ps_attr.name} of chunk_id {chunk_id} to {reset_to_state}"
        )

        # Update the state of tensor and chunk.
        self.chunk_list.update_state(
            chunk_id, param.ps_attr.get_state(access_type), reset_to_state
        )
        param.ps_attr.set_state(reset_to_state, access_type)

        if access_type == AccessType.DATA:
            # NOTE(jiaruifang) device must be the same as the origin param.
            # Or it will affect hook of param.grad_fn.next_functions[0][0].
            param.data = torch.tensor(
                [], dtype=param.ps_attr.data_type, device=param.device
            )
        elif access_type == AccessType.GRAD:
            param.grad = None

        # Check if we finished using all tensors in all chunks of the chunk group,
        # then we can release the remote chunks.
        # The condition for releasing chunks are:
        #     FWD: All non-dummy chunks are of state HOLD_AFTER_FWD;
        #     BWD: All non-dummy chunks are of state HOLD_AFTER_BWD.
        world_size = get_world_size()
        if world_size > 1:
            if with_mem_saving_comm:
                # Check if the chunk_id is ready to reduced or removed.
                # Chunks of diff GPU are in state of HOLD_AFTER_FWD/BWD
                chunk_ready = False
                if training_stage == TrainingStage.FWD:
                    if self.chunk_list[chunk_id].all_tensor_state(
                        TensorState.HOLD_AFTER_FWD
                    ):
                        chunk_ready = True
                elif training_stage == TrainingStage.BWD:
                    if self.chunk_list[chunk_id].all_tensor_state(
                        TensorState.HOLD_AFTER_BWD
                    ):
                        chunk_ready = True

                if chunk_ready:
                    target_rank = -1
                    for cur_rank, cur_chunk_id in enumerate(chunk_id_list):
                        if cur_chunk_id == chunk_id:
                            target_rank = cur_rank
                            break
                    if do_allreduce:
                        # move the chunk_id to GPU
                        self.chunk_list.access_chunk(chunk_id, self.device)
                        if self._time_profile:
                            global_timer.my_timer.start_profile(
                                "CLIENT_release_dist_reduce"
                            )
                        torch.distributed.reduce(
                            self.chunk_list[chunk_id].payload,
                            target_rank,
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False,
                        )
                        if self._time_profile:
                            global_timer.data_move_cnter.update(
                                "CLIENT_release_dist_reduce",
                                self.chunk_list[chunk_id].payload.numel() * 2,
                            )
                            global_timer.my_timer.finish_profile(
                                "CLIENT_release_dist_reduce"
                            )
                        # release chunk payload, only its belonging gpu owns it.
                        if rank == target_rank:
                            self.chunk_list[chunk_id].payload /= world_size
                    if target_rank != rank:
                        self.chunk_list[chunk_id].release_payload()
                        self.set_all_tensors_state_in_chunk(chunk_id, TensorState.FREE)
                    self.visiting_finish(chunk_id)
            else:
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
                    if do_allreduce:
                        if self._time_profile:
                            global_timer.my_timer.start_profile(
                                "CLIENT_release_dist_reduce_scatter"
                            )
                        assert self.chunk_list[local_chunk_id].payload is not None
                        input_list = []
                        for i in chunk_id_list:
                            self.chunk_list.access_chunk(i, self.device)
                            self.chunk_list[i].pin()
                            input_list.append(self.chunk_list[i].payload)
                        torch.distributed.reduce_scatter(
                            self.chunk_list[local_chunk_id].payload,
                            input_list,
                            op=torch.distributed.ReduceOp.SUM,
                            async_op=False,
                        )

                        self.chunk_list[local_chunk_id].payload /= world_size
                        if self._time_profile:
                            global_timer.data_move_cnter.update(
                                "CLIENT_release_dist_reduce_scatter",
                                self.chunk_list[local_chunk_id].payload.numel()
                                * 2
                                * world_size,
                            )
                            global_timer.my_timer.finish_profile(
                                "CLIENT_release_dist_reduce_scatter"
                            )

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
        access_type: AccessType,
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
            access_type: :class:`AccessType`.
            reset_to_state: :class:`TensorState`. The state to reset tensor to.
        """
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return
        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_release")
        rank = self.local_rank
        assert isinstance(reset_to_state, TensorState)

        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        logger.debug(
            f"rank {rank} release a tensor of {access_type} chunk_id {chunk_id} to {reset_to_state}"
        )

        # Update the state of tensor and chunk.
        self.chunk_list.update_state(
            chunk_id, param.ps_attr.get_state(access_type), reset_to_state
        )
        param.ps_attr.set_state(reset_to_state, access_type)

        if access_type == AccessType.DATA:
            # NOTE(jiaruifang) device must be the same as the origin param.
            # Or it will affect hook of param.grad_fn.next_functions[0][0].
            param.data = torch.tensor(
                [], dtype=param.ps_attr.data_type, device=param.device
            )
        elif access_type == AccessType.GRAD:
            param.grad = None

        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_release")

    def release_data(
        self,
        param: torch.nn.Parameter,
        reset_to_state: TensorState = TensorState.HOLD,
    ):
        r"""release the param tensor to FREE or HOLD"""
        self.release(param, AccessType.DATA, reset_to_state)

    def release_grad(
        self,
        param: torch.nn.Parameter,
        reset_to_state: TensorState = TensorState.HOLD,
    ):
        self.release(param, AccessType.GRAD, reset_to_state)

    def reset(self):
        raise NotImplementedError

    def get_overall_chunk_size(self):
        """
        return the overall size of all chunks and
        the overall chunk utilization excluding fragments.
        Excepting the dummy chunk if using MSC.
        """
        overall_size = 0
        overall_chunk_num = 0
        overall_utilization_ratio = 0.0
        for (
            type,
            type_chunk_list,
        ) in self.chunk_tensor_index.chunk_type_to_chunk_id_list_map.items():

            logger.info(f"Chunk list {type}")
            for chunk_id in type_chunk_list:
                chunk = self.chunk_list[chunk_id]
                if self.opt_config["with_mem_saving_comm"] and chunk.is_dummy():
                    continue
                comm_info = self.chunk_tensor_index.chunk_id_to_comm_info_map[chunk_id]
                assert comm_info is not None
                last_used_pos = 0
                for info in self.chunk_tensor_index.generate_tensor_info_in_order(
                    chunk_id
                ):
                    last_used_pos = max(last_used_pos, info.start_offset + info.numel)
                overall_utilization_ratio += last_used_pos / chunk.capacity
                overall_size += chunk.get_chunk_space()
                overall_chunk_num += 1
        overall_utilization_ratio /= overall_chunk_num
        return overall_size, overall_utilization_ratio

    def display_chunk_info(self):
        logger.info("Print chunk list info.")

        overall_size = 0
        overall_chunk_num = 0
        overall_utilization_ratio = 0.0
        for (
            type,
            type_chunk_list,
        ) in self.chunk_tensor_index.chunk_type_to_chunk_id_list_map.items():
            logger.info(f"Chunk list {type}")
            for chunk_id in type_chunk_list:
                chunk = self.chunk_list[chunk_id]
                comm_info = self.chunk_tensor_index.chunk_id_to_comm_info_map[chunk_id]
                assert comm_info is not None

                logger.info(
                    f"Chunk id {chunk.chunk_id}, state {chunk.get_state()}, "
                    f"comm info {comm_info}, "
                    f"capacity {chunk.capacity / 1024 / 1024} M elems, "
                    f"dtype {chunk.data_type} device {chunk.get_device()}"
                )
                last_used_pos = 0
                for info in self.chunk_tensor_index.generate_tensor_info_in_order(
                    chunk_id
                ):
                    assert info.chunk_id == chunk_id, f"{info.chunk_id} vs {chunk_id}"
                    logger.debug(
                        f"** tensor: chunk_id {chunk_id}, start {info.start_offset}, "
                        f"end {info.start_offset + info.numel}, size {info.numel}, "
                        f"tensor_id {info.tensor_id}, state {info.state()}, name {info.tensor_name}"
                    )
                    last_used_pos = max(last_used_pos, info.start_offset + info.numel)
                logger.info(
                    f"chunk used {last_used_pos/1024/1024} M elem, "
                    f"{last_used_pos/chunk.capacity * 100} %"
                )
                overall_utilization_ratio += last_used_pos / chunk.capacity
                overall_size += chunk.get_chunk_space()
                overall_chunk_num += 1

        logger.info(f"OVERALL CHUNK SIZE {overall_size / 1024 / 1024 / 1024} GB")
        logger.info(
            f"OVERALL UTILIZATION {overall_utilization_ratio / overall_chunk_num} %"
        )
