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
import time

from .const import AccessType, PSChunkStatus, PSTensorStatus, ChunkListType
from .chunk_list import ChunkList
import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import logger

from .parameter import PSParameter, is_param_registered


class TensorInfo(object):
    """
    记录chunk内存存储tensor的属性
    PyTorch tensor的存根
    """
    def __init__(self,
                 chunk_id: int,
                 tensor_id: int,
                 start_offset: int,
                 numel: int,
                 param: torch.nn.Parameter,
                 access_type: AccessType,
                 param_name=""):
        self.tensor_id = tensor_id
        self.chunk_id = chunk_id
        self.start_offset = start_offset
        self.numel = numel
        self.param = param
        self.tensor_name = f"{param_name}.data" if (
            access_type == AccessType.DATA) else f"{param_name}.grad"
        self.access_type = access_type
        self.param_fp16_chunk_num = 0

    def status(self):
        """
        访问param中的成员变量很慢
        """
        return self.param.ps_attr.get_status(self.access_type)

    def showme(self):
        logger.info(
            f'tensor_id {self.tensor_id}, name {self.tensor_name}, '
            f'shape {self.param.shape}, chunk_id {self.chunk_id}, '
            f'start_offset {self.start_offset}, nueml {self.numel}, status {self.status()}'
        )


class ChunkTensorIndex(object):
    def __init__(self, default_chunk_size: int = 0):
        """
        一个存储tensor和chunk的检索信息的数据库，每个进程维护一个ChunkTensorIndex实例
        它在预处理阶段根据DNN定义被创建出来
        只有增查功能，无删改
        """
        # 1-1 dict, tensor_id -> TensorInfo
        self.tensor_id_to_info_map: dict[int, TensorInfo] = {}
        # 1-N dict, chunk_id -> List(tensor_id) in order of start_offset
        self.chunk_id_to_tensor_id_list_map: dict[int, List[int]] = {}
        self.chunk_id_to_info_map: dict[int, tuple] = {}

        # (comm_group_idx, chunk_list_type) -> chunk_id_list
        self.comm_group_idx_to_chunk_id_list_map = {}
        # chunk_id -> (comm_group_idx, comm_group_offset, chunk_list_type)
        self.chunk_id_to_comm_group_map = {}

        # 记录不同chunk_list信息，存放chunk_id信息
        self.chunk_type_to_chunk_id_list_map = {}
        self.default_chunk_size = default_chunk_size

    def is_local_chunk(self, chunk_id):
        """
        chunk_id是否是local chunk
        """
        rank = torch.distributed.get_rank()
        grp_id, grp_offset, grp_type = self.chunk_id_to_comm_group_map[
            chunk_id]
        return rank == grp_offset

    def chunk_num(self, list_type: ChunkListType):
        """
        返回chunk_list_type类型chunk list的chunk个数
        """
        if list_type not in self.chunk_type_to_chunk_id_list_map:
            return 0
        else:
            return len(self.chunk_type_to_chunk_id_list_map[list_type])

    def add_chunk(self, chunk_id, chunk_size, data_type, comm_group_id,
                  comm_group_offset, list_type: ChunkListType):
        """
        注册一个chunk信息
        @chunk_id: chunk的id
        @chunk_size: chunk尺寸
        @data_type: chunk中存储数据类型，由于PyTorch限制，再次说明不能在chunk里混合存储两种类型数据，或者将chunk内存以不同类型转换
        @comm_group_id: 在当前list_type中当前通信组的id
        @list_type: chunk做在list的类型
        """
        self.chunk_id_to_info_map[chunk_id] = (chunk_size, data_type)

        comm_group_info = (comm_group_id, list_type)
        if comm_group_info not in self.comm_group_idx_to_chunk_id_list_map:
            self.comm_group_idx_to_chunk_id_list_map[comm_group_info] = list()
        self.comm_group_idx_to_chunk_id_list_map[comm_group_info].append(
            chunk_id)
        self.chunk_id_to_comm_group_map[chunk_id] = (comm_group_id,
                                                     comm_group_offset,
                                                     list_type)

        if list_type not in self.chunk_type_to_chunk_id_list_map:
            self.chunk_type_to_chunk_id_list_map[list_type] = []
        self.chunk_type_to_chunk_id_list_map[list_type].append(chunk_id)

    def get_cur_chunk_num(self):
        return len(self.chunk_id_to_info_map)

    def find_gap(self, numel, data_type):
        """
        在chunk list寻找满足numel大小，类型为data type的空隙
        TODO(jiaruifang) 应该优先分配同设备的gap
        实际使用场景非常具体：在fp16 BWD时，分配grad会在data的chunk内。
        """
        for chunk_id, tensor_info_list in self.chunk_id_to_tensor_id_list_map.items(
        ):
            chunk_size, chunk_data_type = self.chunk_id_to_info_map[chunk_id]
            if chunk_data_type != data_type or chunk_size < numel:
                continue
            prev_end = 0
            for tensor_id in tensor_info_list:
                info = self.tensor_id_to_info_map[tensor_id]
                status = info.status()
                if status == PSTensorStatus.FREE:
                    continue
                start = info.start_offset
                gap = start - prev_end
                if gap >= numel:
                    return chunk_id, prev_end
                prev_end = start + info.numel

            if chunk_size - prev_end >= numel:
                return chunk_id, prev_end

        return None, None

    def reset(self):
        self.tensor_id_to_info_map.clear()
        self.chunk_id_to_tensor_id_list_map.clear()

    def generate_grad_tensor_param(self):
        """
        按chunk内部排列顺序生成所有当前没有被free的grad tensor所在的param
        """
        res_list = []
        for chunk_id, tensor_id_list in self.chunk_id_to_tensor_id_list_map.items(
        ):
            for tensor_id in tensor_id_list:
                info = self.tensor_id_to_info_map[tensor_id]
                if info.access_type == AccessType.GRAD and info.status(
                ) != PSTensorStatus.FREE:
                    res_list.append(info.param)
        return res_list

    def generate_all_tensor_info(self):
        """
        展示每个chunk中tensor的状态
        """
        for chunk_id, tensor_info_list in self.tensor_id_to_info_map.items():
            for tensor_id in tensor_info_list:
                yield self.tensor_id_to_info_map[tensor_id]

    def generate_tensor_info_in_order(self, chunk_id):
        """
        产生在chunk id的所有tensor，以start_offset位置从小到大排序
        O(N)
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
            if self.tensor_id_to_info_map[
                    tensor_id_list[start]].start_offset > start_offset:
                return start
            else:
                return start + 1

        # this occurs if we are moving beyond left\'s boundary
        # meaning the left boundary is the least position to
        # find a number greater than val
        if start > end:
            return start

        mid = (start + end) // 2
        mid_start_offset = self.tensor_id_to_info_map[
            tensor_id_list[mid]].start_offset
        if mid_start_offset < start_offset:
            return self._binary_search(tensor_id_list, start_offset, mid + 1,
                                       end)
        elif mid_start_offset > start_offset:
            return self._binary_search(tensor_id_list, start_offset, start,
                                       mid - 1)
        else:
            return mid

    def add_tensor(self, chunk_id, tensor_id, start_offset, numel, param,
                   access_type):
        """
        添加一个tensor，注册它所属的chunk_id和start_offset信息
        需要将chunk_id内的tensor按照start_offset排序
        二分查找时间复杂度O(logN)
        考虑chunk内排布的tensor可能不连续的情况
        """
        if chunk_id not in self.chunk_id_to_tensor_id_list_map:
            self.chunk_id_to_tensor_id_list_map[chunk_id] = list()

        tensor_id_list = self.chunk_id_to_tensor_id_list_map[chunk_id]
        # 二分查找按照start_offset顺序从小到大插入
        pos = self._binary_search(tensor_id_list, start_offset, 0,
                                  len(tensor_id_list) - 1)
        tensor_id_list.insert(pos, tensor_id)
        if not is_param_registered(param):
            param_name = None
        else:
            param_name = param.ps_attr.name

        self.tensor_id_to_info_map[tensor_id] = TensorInfo(
            chunk_id, tensor_id, start_offset, numel, param, access_type,
            param_name)

        if is_param_registered(param):
            if access_type == AccessType.DATA:
                param.ps_attr.data_chunk_id = chunk_id
            elif access_type == AccessType.GRAD:
                param.ps_attr.grad_chunk_id = chunk_id

    def delete_chunk_id(self, chunk_id):
        """
        @depracated，在静态chunk_schema中不应该被调用
        删除chunk_id对应chunk的索引信息
        """
        raise NotImplementedError
        # 删除chunk中的tensor
        if self.chunk_id_to_tensor_id_list_map.get(chunk_id) is None:
            # logger.info(f'delete_chunk_id {chunk_id} does not exist')
            return
        for tid in self.chunk_id_to_tensor_id_list_map.get(chunk_id, []):
            del self.tensor_id_to_info_map[tid]

        del self.chunk_id_to_tensor_id_list_map[chunk_id]

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

    def get_chunk_id(self, param: torch.nn.Parameter,
                     access_type: AccessType) -> int:
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.tensor_id_to_info_map.get(tensor_id)
        if info is None:
            return None
        return info.chunk_id

    def chunk_ids_of_comm_group(self, chunk_id: int) -> List[int]:
        comm_group_id, _, list_type = self.chunk_id_to_comm_group_map[chunk_id]
        return self.comm_group_idx_to_chunk_id_list_map[(comm_group_id,
                                                         list_type)]

    def generate_all_chunks(self, chunk_list):
        for chunk_id, _ in self.chunk_id_to_tensor_id_list_map.items():
            chunk = chunk_list[chunk_id]
            comm_group_id, comm_group_offset, list_type = self.chunk_id_to_comm_group_map[
                chunk_id]
            yield chunk_id, comm_group_id, chunk

    def visit_chunk(self, chunk):
        rank = torch.distributed.get_rank()
        if rank != 1:
            return
        chunk_id = chunk.chunk_id
        comm_group_id, comm_group_offset, list_type = self.chunk_id_to_comm_group_map[
            chunk_id]
        logger.info(
            f'rank {rank} Chunk id {chunk.chunk_id}, status {chunk.get_status()}, '
            f'comm group ({comm_group_id}, {comm_group_offset}, {list_type}), capacity {chunk.capacity} elems, '
            f'dtype {chunk.data_type}, size {chunk.get_chunk_space()} B, device {chunk.get_device()}'
        )
        for info in self.generate_tensor_info_in_order(chunk_id):
            assert info.chunk_id == chunk_id, f'{info.chunk_id} vs {chunk_id}'
            logger.info(
                f'** tensor: chunk_id {chunk_id}, start {info.start_offset}, '
                f'end {info.start_offset + info.numel}, size {info.numel}, '
                f'tensor_id {info.tensor_id}, status {info.status()}, name {info.tensor_name}'
            )

    def visit_chunks(self, chunk_list: ChunkList):
        rank = torch.distributed.get_rank()
        if rank != 0:
            return
        logger.info(f'visit chunks')

        def print_chunk_list(type_chunk_list):
            total_bytes = 0
            for chunk_id in type_chunk_list:
                chunk = chunk_list[chunk_id]
                comm_group_id, comm_group_offset, list_type = self.chunk_id_to_comm_group_map[
                    chunk_id]
                assert comm_group_id is not None

                logger.info(
                    f'rank {rank} Chunk id {chunk.chunk_id}, status {chunk.get_status()}, '
                    f'comm group {comm_group_id, comm_group_offset, list_type}, capacity {chunk.capacity} elems, '
                    f'dtype {chunk.data_type}, size {chunk.get_chunk_space()} B, device {chunk.get_device()}'
                )
                for info in self.generate_tensor_info_in_order(chunk_id):
                    assert info.chunk_id == chunk_id, f'{info.chunk_id} vs {chunk_id}'
                    logger.info(
                        f'** tensor: chunk_id {chunk_id}, start {info.start_offset}, '
                        f'end {info.start_offset + info.numel}, size {info.numel}, '
                        f'tensor_id {info.tensor_id}, status {info.status()}, name {info.tensor_name}'
                    )
                total_bytes += chunk.get_chunk_space()
            return total_bytes

        overall_size = 0
        overall_size += print_chunk_list(
            self.chunk_type_to_chunk_id_list_map[ChunkListType.PARAM_FP16])

        logger.info(f'OVERALL CHUNK SIZE {overall_size/1e9} GB')

    def _get_tensor_id_list(self, chunk_id):
        if chunk_id not in self.chunk_id_to_tensor_id_list_map:
            self.chunk_id_to_tensor_id_list_map[chunk_id] = list()
        return self.chunk_id_to_tensor_id_list_map[chunk_id]

    def params_generator(self, chunk_id):
        for tensor_id in self.chunk_id_to_tensor_id_list_map[chunk_id]:
            yield self.tensor_id_to_info_map[tensor_id].param

    def try_insert_tensor(self, chunk_id, param, data_type,
                          access_type) -> bool:
        """
        尝试向chunk内插入tensor，返回值表示是否成功
        """
        tensor_id_list = self._get_tensor_id_list(chunk_id)
        prev_end_pos = 0
        assert is_param_registered(param)
        numel = param.ps_attr.numel
        tensor_name = param.ps_attr.name
        target_tensor_id = param.ps_attr.get_tensor_id(access_type)
        for idx, tensor_id in enumerate(tensor_id_list):
            tensor_info = self.tensor_id_to_info_map[tensor_id]
            start_pos = tensor_info.start_offset
            gap = start_pos - prev_end_pos
            if gap >= numel:
                self.tensor_id_to_info_map[target_tensor_id] = TensorInfo(
                    chunk_id, target_tensor_id, prev_end_pos, numel, param,
                    access_type, tensor_name)
                tensor_id_list.insert(idx + 1, target_tensor_id)
                return True
            prev_end_pos = start_pos + tensor_info.numel

        logger.debug(
            f'default_chunk_size {self.default_chunk_size}, prev_end_pos {prev_end_pos}, numel {numel}'
        )
        if self.default_chunk_size - prev_end_pos >= numel:
            self.tensor_id_to_info_map[target_tensor_id] = TensorInfo(
                chunk_id, target_tensor_id, prev_end_pos, numel, param,
                access_type, tensor_name)
            tensor_id_list.insert(len(tensor_id_list), target_tensor_id)
            return True
        return False
