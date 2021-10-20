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
from patrickstar.core.comm import CommInfo

import torch

from patrickstar.utils import logger, get_rank
from .const import AccessType, ChunkType
from .parameter import is_param_registered
from .tensor_stub import TensorInfo


class ChunkTensorIndex(object):
    def __init__(self, default_chunk_size: int = 0):
        r"""
        Storing the index information of tensor and chunks.
        Every process will maintain a `ChunkTensorIndex` instance.
        It is created during preprocessing with the define of the model.
        Only add and search supported, no delete or update.

        Args:
            default_chunk_size: int.
        """
        # 1-1 dict, tensor_id -> TensorInfo
        self.tensor_id_to_info_map: dict[int, TensorInfo] = {}
        # 1-N dict, chunk_id -> List(tensor_id) in order of start_offset
        self.chunk_id_to_tensor_id_list_map: dict[int, List[int]] = {}

        # comm_group -> chunk_id_list
        self.comm_group_to_chunk_id_list_map = {}
        # chunk_id -> comm_info
        self.chunk_id_to_comm_info_map = {}

        # Chunk_ids of chunks in different chunk type
        self.chunk_type_to_chunk_id_list_map = {}
        self.default_chunk_size = default_chunk_size

        # ref_chunk_id -> {chunk_type -> chunk_id}
        self.param_chunk_id_to_os_chunk_id_map = {}

    def register_optimizer_state_chunk_id(
        self,
        ref_param,
        access_type: AccessType,
        chunk_type: ChunkType,
        chunk_id: int,
    ):
        r"""Register the optimizer chunk of type `chunk_type` to `ref_param`.

        Args:
            ref_param: :class:`torch.nn.Parameter`.
            access_type: :class:`AccessType`.
            chunk_type: :class:`ChunkType`.
            chunk_id: int.
        """
        ref_chunk_id = self.get_chunk_id(ref_param, access_type)
        if ref_chunk_id not in self.param_chunk_id_to_os_chunk_id_map:
            self.param_chunk_id_to_os_chunk_id_map[ref_chunk_id] = {
                chunk_type: chunk_id
            }
        else:
            self.param_chunk_id_to_os_chunk_id_map[ref_chunk_id][chunk_type] = chunk_id

    def get_optimizer_state_chunk_id(
        self,
        ref_param: torch.nn.Parameter,
        access_type: AccessType,
        chunk_type: ChunkType,
    ) -> int:
        r"""Get the chunk id storing the optimizer state of `ref_param`.

        Args:
            ref_param: the ref param, usually param fp16
            access_type: :class:`AccessType`.
            chunk_type: type of the optimizer state chunk.
        Returns:
            chunk id, None if not existed.
        """
        ref_chunk_id = self.get_chunk_id(ref_param, access_type)
        if (
            ref_chunk_id not in self.param_chunk_id_to_os_chunk_id_map
            or chunk_type not in self.param_chunk_id_to_os_chunk_id_map[ref_chunk_id]
        ):
            return None
        return self.param_chunk_id_to_os_chunk_id_map[ref_chunk_id][chunk_type]

    def is_local_chunk(self, chunk_id):
        r"""If chunk of `chunk_id` is local.

        Args:
            chunk_id: int.
        Returns:
            bool.
        """
        rank = get_rank()
        comm_info = self.chunk_id_to_comm_info_map[chunk_id]
        return rank == comm_info.offset

    def chunk_num(self, list_type: ChunkType):
        r"""The number of chunks of type `list_type`.

        Args:
            chunk_type: :class:`ChunkType`.
        Returns:
            int.
        """
        if list_type not in self.chunk_type_to_chunk_id_list_map:
            return 0
        else:
            return len(self.chunk_type_to_chunk_id_list_map[list_type])

    def add_chunk(self, chunk_id, comm_info: CommInfo):
        r"""Add a chunk to ChunkTensorIndex.

        Args:
            chunk_id: int.
            comm_info: :class:`CommInfo`.
        """
        comm_group_info = comm_info.group
        if comm_group_info not in self.comm_group_to_chunk_id_list_map:
            self.comm_group_to_chunk_id_list_map[comm_group_info] = list()
        self.comm_group_to_chunk_id_list_map[comm_group_info].append(chunk_id)
        self.chunk_id_to_comm_info_map[chunk_id] = comm_info

        chunk_type = comm_info.chunk_type
        if chunk_type not in self.chunk_type_to_chunk_id_list_map:
            self.chunk_type_to_chunk_id_list_map[chunk_type] = []
        self.chunk_type_to_chunk_id_list_map[chunk_type].append(chunk_id)

    def generate_tensor_info_in_order(self, chunk_id):
        r"""Return the tensors of chunk by `chunk_id`.

        The chunks are ordered by start_offsets.
        """
        for tensor_id in self.chunk_id_to_tensor_id_list_map.get(chunk_id, []):
            yield self.tensor_id_to_info_map[tensor_id]

    def get_tensor_info(self, tensor_id):
        return self.tensor_id_to_info_map[tensor_id]

    def add_tensor(self, chunk_id, tensor_id, start_offset, numel, param, access_type):
        r"""Add a tensor.

        Register the chunk_id of the chunk it belongs and its start_offset in the chunk.
        Support insert tensor between other tensors with binary search.

        TODO(zilinzhu) This method is only called by `append_dummy_chunk`.
        Remove it in the future?
        """
        if chunk_id not in self.chunk_id_to_tensor_id_list_map:
            self.chunk_id_to_tensor_id_list_map[chunk_id] = list()

        tensor_id_list = self.chunk_id_to_tensor_id_list_map[chunk_id]
        tensor_id_list.append(tensor_id)
        if not is_param_registered(param):
            param_name = None
        else:
            param_name = param.ps_attr.name

        self.tensor_id_to_info_map[tensor_id] = TensorInfo(
            chunk_id, tensor_id, start_offset, numel, param, access_type, param_name
        )

        if is_param_registered(param):
            if access_type == AccessType.DATA:
                param.ps_attr.data_chunk_id = chunk_id
            elif access_type == AccessType.GRAD:
                param.ps_attr.grad_chunk_id = chunk_id

    def tensor_id_to_chunk_id(self, tensor_id) -> int:
        r"""Get the chunk id from the tensor id."""
        info = self.tensor_id_to_info_map.get(tensor_id)
        if info is None:
            return None
        return info.chunk_id

    def get_chunk_id(self, param: torch.nn.Parameter, access_type: AccessType) -> int:
        r"""Get the chunk id of the param."""
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        return self.tensor_id_to_chunk_id(tensor_id)

    def chunk_ids_of_comm_group(self, chunk_id: int) -> List[int]:
        comm_info = self.chunk_id_to_comm_info_map[chunk_id]
        return self.comm_group_to_chunk_id_list_map[comm_info.group]

    def _get_tensor_id_list(self, chunk_id):
        if chunk_id not in self.chunk_id_to_tensor_id_list_map:
            self.chunk_id_to_tensor_id_list_map[chunk_id] = list()
        return self.chunk_id_to_tensor_id_list_map[chunk_id]

    def params_generator(self, chunk_id):
        for tensor_id in self.chunk_id_to_tensor_id_list_map[chunk_id]:
            yield self.tensor_id_to_info_map[tensor_id].param

    def delete_tensor(self, chunk_id, param, access_type):
        r"""Delete the tensor from the chunk.

        Args:
            chunk_id: int.
            param: :class:`nn.Parameter`.
            access_type: :class:`AccessType`.
        """
        assert is_param_registered(param)
        target_tensor_id = param.ps_attr.get_tensor_id(access_type)
        if target_tensor_id not in self.tensor_id_to_info_map:
            return
        self.tensor_id_to_info_map.pop(target_tensor_id)
        tensor_id_list = self._get_tensor_id_list(chunk_id)
        tensor_id_list.remove(target_tensor_id)

    def try_insert_tensor_list(self, chunk_id, param_list, access_type):
        r"""Insert a list of param to chunk.

        Notice that this method is an atomic method: if we failed to insert
        any param of the list, we will delete all the params in the list that
        are already inserted.

        Args:
            chunk_id: int.
            param_list: list of :class:`nn.Parameter`.
            access_type: :class:`AccessType`.
        Returns:
            Whether the insertion was successful.
        """
        visited_params = []
        success = True
        for param in param_list:
            success = self.try_insert_tensor(chunk_id, param, access_type)
            visited_params.append(param)
            if not success:
                break
        if not success:
            for param in visited_params:
                self.delete_tensor(chunk_id, param, access_type)

        return success

    def try_insert_tensor(self, chunk_id, param, access_type) -> bool:
        r"""
        Try inserting tensor to chunk, return successful or not.
        If `param` was inserted, return True.

        Args:
            chunk_id: int.
            param: :class:`nn.Parameter`.
            access_type: :class:`AccessType`.
        Returns:
            Whether the insertion was successful.
        """
        tensor_id_list = self._get_tensor_id_list(chunk_id)
        prev_end_pos = 0
        assert is_param_registered(param)
        numel = param.ps_attr.numel
        tensor_name = param.ps_attr.name
        target_tensor_id = param.ps_attr.get_tensor_id(access_type)
        for idx, tensor_id in enumerate(tensor_id_list):
            if target_tensor_id == tensor_id:
                return True
        for idx, tensor_id in enumerate(tensor_id_list):
            tensor_info = self.tensor_id_to_info_map[tensor_id]
            start_pos = tensor_info.start_offset
            gap = start_pos - prev_end_pos
            if gap >= numel:
                self.tensor_id_to_info_map[target_tensor_id] = TensorInfo(
                    chunk_id,
                    target_tensor_id,
                    prev_end_pos,
                    numel,
                    param,
                    access_type,
                    tensor_name,
                )
                tensor_id_list.insert(idx + 1, target_tensor_id)
                return True
            prev_end_pos = start_pos + tensor_info.numel

        logger.debug(
            f"default_chunk_size {self.default_chunk_size}, prev_end_pos {prev_end_pos}, numel {numel}"
        )
        if self.default_chunk_size - prev_end_pos >= numel:
            self.tensor_id_to_info_map[target_tensor_id] = TensorInfo(
                chunk_id,
                target_tensor_id,
                prev_end_pos,
                numel,
                param,
                access_type,
                tensor_name,
            )
            tensor_id_list.insert(len(tensor_id_list), target_tensor_id)
            return True
        return False
