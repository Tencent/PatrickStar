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

from typing import List

import torch

from patrickstar.utils import logger, get_rank
from .const import AccessType, ChunkType
from .parameter import is_param_registered
from .tensor_stub import TensorInfo


class ChunkTensorIndex(object):
    def __init__(self, default_chunk_size: int = 0):
        """
        Storing the index information of tensor and chunks.
        Every process will maintain a `ChunkTensorIndex` instance.
        It is created during preprocessing with the define of the model.
        Only add and search supported, no delete or update.
        """
        # 1-1 dict, tensor_id -> TensorInfo
        self.tensor_id_to_info_map: dict[int, TensorInfo] = {}
        # 1-N dict, chunk_id -> List(tensor_id) in order of start_offset
        self.chunk_id_to_tensor_id_list_map: dict[int, List[int]] = {}

        # (comm_group_idx, chunk_type) -> chunk_id_list
        self.comm_group_idx_to_chunk_id_list_map = {}
        # chunk_id -> (comm_group_idx, comm_group_offset, chunk_type)
        self.chunk_id_to_comm_group_map = {}

        # Chunk_ids of chunks in different chunk type
        self.chunk_type_to_chunk_id_list_map = {}
        self.default_chunk_size = default_chunk_size

        # key is a tuple (torch.nn.Parameter, ParamListType) -> value: int
        self.param_fp16_chunk_id_to_os_chunk_id_map = {}

    def register_optimizer_state_chunk_id(
        self,
        ref_param,
        access_type: AccessType,
        chunk_type: ChunkType,
        chunk_id: int,
    ):
        ref_chunk_id = self.get_chunk_id(ref_param, access_type)
        self.param_fp16_chunk_id_to_os_chunk_id_map[
            (ref_chunk_id, chunk_type)
        ] = chunk_id

    def get_optimizer_state_chunk_id(
        self,
        ref_param: torch.nn.Parameter,
        access_type: AccessType,
        chunk_type: ChunkType,
    ) -> int:
        """
        Get the chunk id storing the optimizer state of `ref_param`.
        args:
            @ref_param: the ref param, usually param fp16
            @access_type: AccessType
            @chunk_type: type of the optimizer state chunk.
        rets:
            chunk id, None if not existed.
        """
        ref_chunk_id = self.get_chunk_id(ref_param, access_type)
        return self.param_fp16_chunk_id_to_os_chunk_id_map.get(
            (ref_chunk_id, chunk_type)
        )

    def is_local_chunk(self, chunk_id):
        """
        If chunk of `chunk_id` is local.
        """
        rank = get_rank()
        _, grp_offset, _ = self.chunk_id_to_comm_group_map[chunk_id]
        return rank == grp_offset

    def chunk_num(self, list_type: ChunkType):
        """
        The number of chunks of type `list_type`.
        """
        if list_type not in self.chunk_type_to_chunk_id_list_map:
            return 0
        else:
            return len(self.chunk_type_to_chunk_id_list_map[list_type])

    def add_chunk(
        self, chunk_id, comm_group_id, comm_group_offset, list_type: ChunkType
    ):
        comm_group_info = (comm_group_id, list_type)
        if comm_group_info not in self.comm_group_idx_to_chunk_id_list_map:
            self.comm_group_idx_to_chunk_id_list_map[comm_group_info] = list()
        self.comm_group_idx_to_chunk_id_list_map[comm_group_info].append(chunk_id)
        self.chunk_id_to_comm_group_map[chunk_id] = (
            comm_group_id,
            comm_group_offset,
            list_type,
        )

        if list_type not in self.chunk_type_to_chunk_id_list_map:
            self.chunk_type_to_chunk_id_list_map[list_type] = []
        self.chunk_type_to_chunk_id_list_map[list_type].append(chunk_id)

    def generate_tensor_info_in_order(self, chunk_id):
        """
        Return the tensors of chunk by `chunk_id`.
        The chunks are ordered by start_offsets.
        """
        for tensor_id in self.chunk_id_to_tensor_id_list_map.get(chunk_id, []):
            yield self.tensor_id_to_info_map[tensor_id]

    def get_tensor_info(self, tensor_id):
        return self.tensor_id_to_info_map[tensor_id]

    def _binary_search(self, tensor_id_list, start_offset, start, end):
        # we need to distinugish whether we should insert
        # before or after the left boundary.
        # imagine [0] is the last step of the binary search
        # and we need to decide where to insert -1
        if start == end:
            if (
                self.tensor_id_to_info_map[tensor_id_list[start]].start_offset
                > start_offset
            ):
                return start
            else:
                return start + 1

        # this occurs if we are moving beyond left\'s boundary
        # meaning the left boundary is the least position to
        # find a number greater than val
        if start > end:
            return start

        mid = (start + end) // 2
        mid_start_offset = self.tensor_id_to_info_map[tensor_id_list[mid]].start_offset
        if mid_start_offset < start_offset:
            return self._binary_search(tensor_id_list, start_offset, mid + 1, end)
        elif mid_start_offset > start_offset:
            return self._binary_search(tensor_id_list, start_offset, start, mid - 1)
        else:
            return mid

    def add_tensor(self, chunk_id, tensor_id, start_offset, numel, param, access_type):
        """
        Add a tensor.
        Register the chunk_id of the chunk it belongs and its start_offset in the chunk.
        Support insert tensor between other tensors with binary search.
        """
        if chunk_id not in self.chunk_id_to_tensor_id_list_map:
            self.chunk_id_to_tensor_id_list_map[chunk_id] = list()

        tensor_id_list = self.chunk_id_to_tensor_id_list_map[chunk_id]
        # 二分查找按照start_offset顺序从小到大插入
        pos = self._binary_search(
            tensor_id_list, start_offset, 0, len(tensor_id_list) - 1
        )
        tensor_id_list.insert(pos, tensor_id)
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
        """
        tensor_id -> chunk_id
        """
        # info =
        info = self.tensor_id_to_info_map.get(tensor_id)
        if info is None:
            return None
        else:
            return info.chunk_id

    def get_chunk_id(self, param: torch.nn.Parameter, access_type: AccessType) -> int:
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.tensor_id_to_info_map.get(tensor_id)
        if info is None:
            return None
        return info.chunk_id

    def chunk_ids_of_comm_group(self, chunk_id: int) -> List[int]:
        comm_group_id, _, list_type = self.chunk_id_to_comm_group_map[chunk_id]
        return self.comm_group_idx_to_chunk_id_list_map[(comm_group_id, list_type)]

    def generate_all_chunks(self, chunk_list):
        for chunk_id, _ in self.chunk_id_to_tensor_id_list_map.items():
            chunk = chunk_list[chunk_id]
            comm_group_id, _, _ = self.chunk_id_to_comm_group_map[chunk_id]
            yield chunk_id, comm_group_id, chunk

    def _get_tensor_id_list(self, chunk_id):
        if chunk_id not in self.chunk_id_to_tensor_id_list_map:
            self.chunk_id_to_tensor_id_list_map[chunk_id] = list()
        return self.chunk_id_to_tensor_id_list_map[chunk_id]

    def params_generator(self, chunk_id):
        for tensor_id in self.chunk_id_to_tensor_id_list_map[chunk_id]:
            yield self.tensor_id_to_info_map[tensor_id].param

    def delete_tensor(self, chunk_id, param, access_type):
        assert is_param_registered(param)
        target_tensor_id = param.ps_attr.get_tensor_id(access_type)
        if target_tensor_id not in self.tensor_id_to_info_map:
            return
        self.tensor_id_to_info_map.pop(target_tensor_id)
        tensor_id_list = self._get_tensor_id_list(chunk_id)
        tensor_id_list.remove(target_tensor_id)

    def try_insert_tensor_list(self, chunk_id, param_list, access_type):
        """
        Insert a list of param to chunk.
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
        """
        Try inserting tensor to chunk, return successful or not.
        If `param` was inserted, return True.
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
