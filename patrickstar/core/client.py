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

import os
from typing import List

import torch

import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import logger, get_world_size, get_rank
from .chunk_list import ChunkList, ChunkListType
from .chunk_tensor_index import ChunkTensorIndex
from .const import AccessType, PSChunkStatus, PSTensorStatus, TrainingStage
from .hook import setup_patrickstar_hooks
from .parameter import register_param, is_param_registered, ParamType


class PatrickStarClient(object):
    def __init__(self, rank: int, default_chunk_size: int):
        """
        管理一个Process的Param, AccGrad, OS数据。
        每个进程可以访问一个GPU的显存，和cpu的内存
        功能:
          1. 充分利用cpu和gpu内存
          2. 细粒度调度，PatrickStarClient包含若干chunk
        """
        self.pid = os.getpid()

        # index of gpu
        self.local_rank = rank

        self.module = None

        # 解耦chunk和param的tensors
        self.default_chunk_size = default_chunk_size

        self.chunk_tensor_index = ChunkTensorIndex(self.default_chunk_size)
        self.chunk_list = ChunkList(self.local_rank)

        self._time_profile = True

        self._chunk_id = -1

        if torch.distributed.is_initialized():
            self.cpu_comm_group = torch.distributed.new_group(backend="gloo")
        else:
            self.cpu_comm_group = None

        self.dummy_param_list = []
        self.torch_param_list = []
        self.param_fp16_to_param_fp32_map = {}
        self.chunk_based_param_fp16 = []

        # for post backward hook
        self.grad_accs = []

    def _generate_chunk_id(self):
        self._chunk_id += 1
        return self._chunk_id

    def init(self, model, optimizer):
        """
        初始化client管理的model和optimizer
        并完成model的参数的分配和拷贝
        """
        self.module = model
        self.optimizer = optimizer
        if get_rank() == 0:
            self.display_chunk_info()
        self.register_model_hook(model)

    def append_dummy_chunk(
        self, data_type: torch.dtype, chunk_list_type: ChunkListType
    ):
        """
        向chunk_list_type list中添加一个dummy chunk
        """
        tmp_chunk_id = self.chunk_list.generate_chunk_id()
        comm_group_idx, comm_group_offset = self.chunk_list.new_chunk(
            tmp_chunk_id,
            self.default_chunk_size,
            torch.half,
            is_dummy=True,
            chunk_type=chunk_list_type,
        )
        self.chunk_tensor_index.add_chunk(
            tmp_chunk_id,
            comm_group_idx,
            comm_group_offset,
            chunk_list_type,
        )
        dummy = torch.nn.Parameter(
            torch.tensor([], dtype=data_type), requires_grad=False
        )
        # 加入一个dummy param可以让dummy chunk状态被设置为hold
        register_param(
            dummy, ParamType.CHUNK_BASED, torch.half, f"dummy_{comm_group_idx}"
        )
        self.dummy_param_list.append(dummy)
        self.chunk_tensor_index.add_tensor(
            tmp_chunk_id,
            self.dummy_param_list[-1].ps_attr.data_id(),
            0,
            dummy.numel(),
            self.dummy_param_list[-1],
            AccessType.DATA,
        )

        logger.info(
            f"Append a dummy chunk to the Chunk List {chunk_list_type} "
            f"comm group ({comm_group_idx} {comm_group_offset})"
        )

    def delete_param(self, param, access_type):
        """
        TODO(jiaruifang) Remove tensor of the param
        """
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, AccessType.DATA)
        self.chunk_tensor_index.delete_tensor(chunk_id, param, AccessType.DATA)

    def append_tensor(
        self,
        param_list: List[torch.nn.Parameter],
        data_type: torch.dtype,
        access_type: AccessType,
        chunk_list_type: ChunkListType,
    ):
        """
        将一个tensor交给client管理，这个tensor必须是某个parameter的data或者grad成员变量
        具体过程，如果这个param之前没有被client管理过，则在对应的chunk_list_type后append这个tensor
        args:
            @param_list: tensor list所在的Parameter list，client管理的tensor必须属于parameter的data或者grad
            @data_type: tensor的数据类型，可以和param的类型不一致，因为param后面会被改变类型
            @access_type: 访问data或者grad
            @chunk_list_type: tensor插入队列的类型
        """
        assert type(data_type) == torch.dtype, f"data_type is {type(data_type)}"
        assert type(access_type) == AccessType
        if self.chunk_list.is_empty(chunk_list_type):
            chunk_id = self.chunk_list.generate_chunk_id()
        else:
            last_chunk_id = self.chunk_list.last_chunk_id(chunk_list_type)
            is_success = self.chunk_tensor_index.try_insert_tensor_list(
                last_chunk_id, param_list, access_type
            )
            if is_success:
                return
            chunk_id = self.chunk_list.generate_chunk_id()

        comm_group_idx, comm_group_offset = self.chunk_list.new_chunk(
            chunk_id, self.default_chunk_size, data_type, False, chunk_list_type
        )
        self.chunk_tensor_index.add_chunk(
            chunk_id,
            comm_group_idx,
            comm_group_offset,
            chunk_list_type,
        )
        is_success = self.chunk_tensor_index.try_insert_tensor_list(
            chunk_id, param_list, access_type
        )
        if not is_success:
            raise RuntimeError(
                f"can not append a tensor to chunk_tensor_index."
                f"Overall size of param list is larger than the default chunk size {self.default_chunk_size}."
            )
        return

    def append_tensor_as_ref(
        self, param, data_type, access_type, chunk_list_type, ref_param
    ):
        """
        按照ref_param的排布方式排布param
        ref_param和param所在的chunk id不同
        """
        chunk_id = self.chunk_tensor_index.get_optimizer_state_chunk_id(
            ref_param, access_type, chunk_list_type
        )
        if chunk_id is None:
            # new a chunk
            chunk_id = self.chunk_list.generate_chunk_id()
            comm_group_idx, comm_group_offset = self.chunk_list.new_chunk(
                chunk_id, self.default_chunk_size, data_type, False, chunk_list_type
            )
            self.chunk_tensor_index.add_chunk(
                chunk_id,
                comm_group_idx,
                comm_group_offset,
                chunk_list_type,
            )
        ret = self.chunk_tensor_index.try_insert_tensor(chunk_id, param, access_type)
        self.chunk_tensor_index.register_optimizer_state_chunk_id(
            ref_param, access_type, chunk_list_type, chunk_id
        )
        assert ret is True

    def param_fp16_chunks_max_mem_usage(self):
        """
        获得param fp16使用Chunk所占的内存大小 (in Bytes)
        在多机环境，需要包括allgather获得remote chunks
        """
        world_size = get_world_size()
        # 本进程自己管理的Chunk，和Group Chunk Buff会分配的Chunk
        return (
            self.chunk_tensor_index.chunk_num(ChunkListType.PARAM_FP16)
            * self.default_chunk_size
            * 2
            / world_size
            + (world_size - 1) * self.default_chunk_size * 2
        )

    def set_all_tensors_status_in_chunk(self, chunk_id, new_status):
        """
        把一个chunk所有的tensor状态设置为status，chunk的状态也随之改变
        不管payload是否被分配
        """
        for info in self.chunk_tensor_index.generate_tensor_info_in_order(chunk_id):
            param = info.param
            access_type = info.access_type
            old_status = param.ps_attr.get_status(access_type)
            self.chunk_list.update_status(chunk_id, old_status, new_status)
            param.ps_attr.set_status(new_status, access_type)

    def register_model_hook(self, model):
        setup_patrickstar_hooks(model, self)

    def chunk_ids_generator(self, chunk_list_type: ChunkListType):
        return self.chunk_list.chunk_ids_generator(chunk_list_type)

    def is_local_param(self, param, access_type):
        """Check if param is in local chunk"""
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        return self.chunk_tensor_index.is_local_chunk(chunk_id)

    def _fetch_remote_chunks(
        self,
        chunk_id_list,
        local_chunk_id,
        compute_device,
        param_name,
        training_stage: TrainingStage,
    ):
        """
        将chunk_id_list中远端的chunk取到本地
        """
        rank = get_rank()

        # FWD过程，当global chunk中有param第一次被访问时，需要将global chunk收集到本地。
        # 如何判断global chunk中有param第一次被访问的时刻，从而正确触发allgather操作。
        # 第一个param被访问时的必要条件是remote chunk状态为RELEASED。
        # 因此，当每个chunk由HOLD_AFTER_FWD(HOLD_ADFTER_BWD)->RELEASED时
        has_released_chunk = False
        for i in chunk_id_list:
            if self.chunk_list[i].get_status() == PSChunkStatus.RELEASED:
                has_released_chunk = True
                break
        if not has_released_chunk:
            return

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_fetch_remote_chunks")

        logger.debug(
            f"rank {rank} fetch {param_name} remote chunks {chunk_id_list} local chunk {local_chunk_id}"
        )
        allgather_payload_buff = []

        local_chunk_payload = None
        comm_data_amount = 0
        for chunk_id in chunk_id_list:
            if chunk_id == local_chunk_id:
                local_chunk_payload = self.chunk_list[chunk_id].payload
                allgather_payload_buff.append(local_chunk_payload)
            else:
                self.chunk_list.prepare_device(
                    compute_device, self.chunk_list[chunk_id].get_chunk_space()
                )
                # TODO(jiaruifang) 此处可以不分配空间，用一个复用的comm_buffer
                self.chunk_list[chunk_id].allocate_payload(compute_device)
                # 刚分配的chunk，以备allgather使用，allgather之前不要被换出。
                self.chunk_list[chunk_id].pin()
                self.set_all_tensors_status_in_chunk(chunk_id, PSTensorStatus.HOLD)
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
            global_timer.my_timer.start_profile("CLIENT_fetch_remote_chunks_allgather")

        logger.info(f"rank {rank} allgather {chunk_id_list}")
        torch.distributed.all_gather(
            allgather_payload_buff, local_chunk_payload, async_op=False
        )

        allgather_payload_buff = []
        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_fetch_remote_chunks_allgather")
            global_timer.data_move_cnter.update(
                "CLIENT_fetch_remote_chunks_allgather", comm_data_amount
            )
            global_timer.my_timer.finish_profile("CLIENT_fetch_remote_chunks")

    def access_dist(
        self,
        param: torch.nn.Parameter,
        access_type: AccessType,
        compute_device: torch.device,
        training_stage: TrainingStage,
    ) -> torch.Tensor:
        """
        在分布式训练场景下，访问param的tensor，串行也可以使用
        @args
            param: 待访问的参数
            access_type: 访问方式
            compute_device: 目标设备
            training_stage: 训练解阶段
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

        # 准备param所在chunk的内存，如果内存不在计算设备上需要分配或者移动
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        chunk_id_list = self.chunk_tensor_index.chunk_ids_of_comm_group(chunk_id)
        rank = get_rank()

        if get_world_size() > 1:
            local_chunk_id = chunk_id_list[rank]

            logger.debug(
                f"rank {rank} access_dist access tensor {param.ps_attr.name} "
                f"local_chunk_id {local_chunk_id} chunk_id_list {chunk_id_list}"
            )

            # 每个进程把local_chunk_id都弄到本地
            self.chunk_list.access_chunk(local_chunk_id, compute_device)

            # _fetch_remote_chunks不要将local_chunk_id也给换出去了，
            # 因为它的状态还是HOLD，加上pin。
            self.chunk_list[local_chunk_id].pin()

            self._fetch_remote_chunks(
                chunk_id_list,
                local_chunk_id,
                compute_device,
                param.ps_attr.name,
                training_stage,
            )
            self.chunk_list[local_chunk_id].unpin()

        # _fetch_remote_chunks可能不执行allgather，此时远端的chunk在本地，需要取到计算设备上。
        self.chunk_list.access_chunk(chunk_id, compute_device)

        # 将param内存定位到chunk上
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.chunk_tensor_index.get_tensor_info(tensor_id)
        start_offset = info.start_offset
        numel = info.numel
        assert numel == param.ps_attr.numel, f"{numel} vs {param.ps_attr.numel}"

        assert (
            self.chunk_list[chunk_id].payload is not None
        ), f"rank {rank} chunk id {chunk_id}' payload is None'"
        assert self.chunk_list[chunk_id].payload.device == compute_device, (
            f"rank {rank} chunk id {chunk_id}' payload is not on "
            f"{compute_device}, but on "
            f"{self.chunk_list[chunk_id].payload.device}"
        )

        param.ps_attr.set_tensor(
            self.chunk_list[chunk_id].payload.narrow(0, start_offset, numel),
            access_type,
        )

        # 改变param's tensor对应chunk的status，chunk状态由它管理的所有tensor状态共同决定。
        old_status = param.ps_attr.get_status(access_type)

        # 如果是从free/uninit状态转换的需要清零
        if old_status == PSTensorStatus.FREE:
            param.ps_attr.access_tensor(access_type).zero_()

        # 访问之后应该更新Tensor的状态，鉴于chunk状态是由它管理tensor共同决定
        # 因此tensor对应的chunk的状态随之改变
        # dist情况
        self.chunk_list.update_status(chunk_id, old_status, PSTensorStatus.COMPUTE)
        param.ps_attr.set_status(PSTensorStatus.COMPUTE, access_type)

        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_access_dist")

        return param.ps_attr.access_tensor(access_type)

    def access(
        self,
        param: torch.nn.Parameter,
        access_type: AccessType,
        compute_device: torch.device,
    ):
        """
        访问`nn.Parameter`的data或者grad，让它们参与计算。
        具体步骤
        找到param的tensor对应的chunk。
        1. 如果chunk的payload存在
        然后决定是否移动chunk到计算设备，移动之前要给计算设备腾出足够空间。
        2. 如果chunkd的payload不存在
        比如grad FP16，在step过程所在的chunk已经被标记为FREE，并被释放掉。
        将payload分配到计算设备的内存上，分配前需要给计算设备腾出足够空间。

        异常：一个chunk中两个tensor的计算设备不一致。
        两种access，
        1. FWD+BWD过程的access_data and access_grad?
        如果访问的是本地chunk_id，也需要allgather配合其他process
        2. adam过程access_data
        只能访问本地chunk不需要通信
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

        # 准备param所在chunk的内存，如果内存不在计算设备上需要分配或者移动
        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)

        # 这个tensor还没在chunk schema中
        if chunk_id is None:
            raise RuntimeError(
                "FP16 training shall not meet tensors with no chunk assigned. "
                "Every tensor has to be assigned to a chunk during a tensor-chunk-mapping "
                "process before training."
            )

        self.chunk_list.access_chunk(chunk_id, compute_device)

        # 将param内存定位到chunk上
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.chunk_tensor_index.get_tensor_info(tensor_id)
        start_offset = info.start_offset
        numel = info.numel
        assert numel == param.ps_attr.numel, f"{numel} vs {param.ps_attr.numel}"

        param.ps_attr.set_tensor(
            self.chunk_list[chunk_id].payload.narrow(0, start_offset, numel),
            access_type,
        )

        old_status = param.ps_attr.get_status(access_type)

        # 如果是从free状态转换的需要清零，或者从
        if old_status == PSTensorStatus.FREE:
            param.ps_attr.access_tensor(access_type).zero_()

        # 访问之后应该更新Tensor的状态，chunk的状态随之改变
        self.chunk_list.update_status(chunk_id, old_status, PSTensorStatus.COMPUTE)
        param.ps_attr.set_status(PSTensorStatus.COMPUTE, access_type)

        # Note并不设置parameter对应的tensor，因为adam可能直接访问pstensor
        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_access")
        return param.ps_attr.access_tensor(access_type)

    def access_data(self, param: torch.nn.Parameter, compute_device: torch.device):
        """
        将param的ps_data_tensor的数据放置到compute_device上
        """
        return self.access(param, AccessType.DATA, compute_device)

    def access_grad(self, param: torch.nn.Parameter, compute_device: torch.device):
        """
        将param的ps_grad_tensor的数据放置到compute_device上
        NOTE，并没有正确设置param的grad，此时grad的数据无效。因为grad的设备属性并不自由，
        需要看data的脸色行事。我们使用grad时候，需要显式设置
        `param.grad = param.ps_grad_tensore`
        """
        return self.access(param, AccessType.GRAD, compute_device)

    def release_dist(
        self,
        param: torch.nn.Parameter,
        access_type: AccessType,
        reset_to_status: PSTensorStatus,
        training_stage: TrainingStage,
        is_allreduce: bool,
    ):
        """
        这个param的data, grad不再需要放在计算设备
        1. 更新状态
        首先更新tensor和chunk的状态
        2. 释放内存
        在释放Parameter中tensor的内存，释放PSTensor中的内存
        看看是否有chunk的状态为free，释放chunk内存
        """
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return

        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_release_dist")
        rank = get_rank()

        assert isinstance(reset_to_status, PSTensorStatus)
        assert (
            training_stage == TrainingStage.FWD or training_stage == TrainingStage.BWD
        )
        assert torch.distributed.is_initialized()

        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        # 可以在tensor-chunk schema构造过程中获得local_chunk_id
        chunk_id_list = self.chunk_tensor_index.chunk_ids_of_comm_group(chunk_id)

        local_chunk_id = chunk_id_list[rank]

        logger.debug(
            f"rank {rank} release tensor {param.ps_attr.name} of chunk_id {chunk_id} to {reset_to_status}"
        )

        # 更新tensor和chunk状态， tensor被设置为free，需要删除内存
        # 释放tensor的内存，再释放chunk内存
        self.chunk_list.update_status(
            chunk_id, param.ps_attr.get_status(access_type), reset_to_status
        )
        param.ps_attr.set_status(reset_to_status, access_type)

        # 找到需要删除的chunk，先删除chunk关联的tensors
        if access_type == AccessType.DATA:
            # NOTE(jiaruifang) 必须device和原来param一致，影响hook of param.grad_fn.next_functions[0][0]
            param.data = torch.tensor(
                [], dtype=param.ps_attr.data_type, device=param.device
            )
        elif access_type == AccessType.GRAD:
            param.grad = None

        # 判断chunk group中所有的chunk都被使用完毕，可以释放remote chunk
        # FWD: 当所有非dummy的chunk都是HOLD_AFTER_FWD
        # BWD: 当所有非dummy的chunk都是HOLD_AFTER_BWD
        world_size = get_world_size()
        if world_size > 1:
            all_chunks_ready = True
            for i in chunk_id_list:
                if training_stage == TrainingStage.FWD:
                    if (
                        not self.chunk_list[i].all_tensor_status(
                            PSTensorStatus.HOLD_AFTER_FWD
                        )
                        and not self.chunk_list[i].is_dummy()
                    ):
                        all_chunks_ready = False
                elif training_stage == TrainingStage.BWD:
                    if (
                        not self.chunk_list[i].all_tensor_status(
                            PSTensorStatus.HOLD_AFTER_BWD
                        )
                        and not self.chunk_list[i].is_dummy()
                    ):
                        all_chunks_ready = False

            if all_chunks_ready:
                if is_allreduce:
                    if self._time_profile:
                        global_timer.my_timer.start_profile(
                            "CLIENT_release_dist_reduce_scatter"
                        )
                    assert self.chunk_list[local_chunk_id].payload is not None
                    input_list = []
                    for i in chunk_id_list:
                        self.chunk_list.access_chunk(
                            i, torch.device(f"cuda:{self.local_rank}")
                        )
                        self.chunk_list[i].pin()
                        input_list.append(self.chunk_list[i].payload)
                    torch.distributed.reduce_scatter(
                        self.chunk_list[local_chunk_id].payload,
                        input_list,
                        op=torch.distributed.ReduceOp.SUM,
                        async_op=False,
                    )

                    # NOTE把下面行注释了不影响最终结果？loss可能是有softmax算出，所以相对值不影响LOSS比较，但是影响了
                    # 不应该除以world_size,减去dummy chunk个数
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

                # 删除remote chunk的payload
                for i in chunk_id_list:
                    self.chunk_list[i].unpin()
                    if i != local_chunk_id:
                        logger.debug(f"rank {rank} remove payload of chunk_id {i}")
                        self.chunk_list[i].release_payload()
                        self.set_all_tensors_status_in_chunk(i, PSTensorStatus.FREE)

        if self._time_profile:
            global_timer.my_timer.finish_profile("CLIENT_release_dist")

    def release(
        self,
        param: torch.nn.Parameter,
        access_type: AccessType,
        reset_to_status: PSTensorStatus = PSTensorStatus.HOLD,
    ):
        """
        这个param的data, grad不再需要放在计算设备
        1. 更新状态
        首先更新tensor和chunk的状态
        2. 释放内存
        在释放Parameter中tensor的内存，释放PSTensor中的内存
        看看是否有chunk的状态为free，释放chunk内存
        """
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            return
        if self._time_profile:
            global_timer.my_timer.start_profile("CLIENT_release")
        rank = self.local_rank
        assert isinstance(reset_to_status, PSTensorStatus)

        chunk_id = self.chunk_tensor_index.get_chunk_id(param, access_type)
        logger.debug(
            f"rank {rank} release a tensor of {access_type} chunk_id {chunk_id} to {reset_to_status}"
        )

        # 更新tensor和chunk状态，如果tensor被设置为free，需要删除ps_tensor的内存
        self.chunk_list.update_status(
            chunk_id, param.ps_attr.get_status(access_type), reset_to_status
        )
        param.ps_attr.set_status(reset_to_status, access_type)

        # 找到需要删除的chunk，先删除chunk关联的tensors
        if access_type == AccessType.DATA:
            # NOTE() 必须to device它和param.grad_fn.next_functions[0][0]
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
        reset_to_status: PSTensorStatus = PSTensorStatus.HOLD,
    ):
        """
        可以把一个tensor释放成FREE，也可以成HOLD
        """
        self.release(param, AccessType.DATA, reset_to_status)

    def release_grad(
        self,
        param: torch.nn.Parameter,
        reset_to_status: PSTensorStatus = PSTensorStatus.HOLD,
    ):
        self.release(param, AccessType.GRAD, reset_to_status)

    def reset(self):
        """
        删除chunk_list和chunk_tensor_index
        """
        raise NotImplementedError

    def display_chunk_info(self):
        logger.info("Print chunk list info.")

        overall_size = 0
        for (
            type,
            type_chunk_list,
        ) in self.chunk_tensor_index.chunk_type_to_chunk_id_list_map.items():
            logger.info(f"Chunk list {type}")
            for chunk_id in type_chunk_list:
                chunk = self.chunk_list[chunk_id]
                (
                    comm_group_id,
                    comm_group_offset,
                    _,
                ) = self.chunk_tensor_index.chunk_id_to_comm_group_map[chunk_id]
                assert comm_group_id is not None

                logger.info(
                    f"Chunk id {chunk.chunk_id}, status {chunk.get_status()}, "
                    f"comm group {comm_group_id, comm_group_offset}, "
                    f"capacity {chunk.capacity / 1024 / 1024} M elems, "
                    f"dtype {chunk.data_type} device {chunk.get_device()}"
                )
                for info in self.chunk_tensor_index.generate_tensor_info_in_order(
                    chunk_id
                ):
                    assert info.chunk_id == chunk_id, f"{info.chunk_id} vs {chunk_id}"
                    logger.debug(
                        f"** tensor: chunk_id {chunk_id}, start {info.start_offset}, "
                        f"end {info.start_offset + info.numel}, size {info.numel}, "
                        f"tensor_id {info.tensor_id}, status {info.status()}, name {info.tensor_name}"
                    )
                overall_size += chunk.get_chunk_space()

        logger.info(f"OVERALL CHUNK SIZE {overall_size / 1024 / 1024 / 1024} GB")
