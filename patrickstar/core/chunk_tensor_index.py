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

from .const import PSChunkStatus, PSTensorStatus, ChunkListType
from .chunk_list import ChunkList
import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import logger

from .parameter import PSParameter, is_param_registed


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
                 param_name=""):
        self.tensor_id = tensor_id
        self.chunk_id = chunk_id
        self.start_offset = start_offset
        self.numel = numel
        self.param = param
        self.tensor_name = f"{param_name}"
        self.param_fp16_chunk_num = 0

    def status(self):
        """
        访问param中的成员变量很慢
        """
        return self.param.ps_attr.status()

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
        self.dict_tensor_id_info: dict[int, TensorInfo] = {}
        # 1-N dict, chunk_id -> List(tensor_id) in order of start_offset
        self.dict_chunk_id_tensor_id: dict[int, List[int]] = {}
        self.dict_chunk_id_chunk_info: dict[int, tuple] = {}

        # comm_group_id 对应的chunk_id_list
        self.comm_group_id_chunk_id_list = {}
        self.dict_chunk_id_comm_group_id = {}

        # 记录不同chunk_list信息，存放chunk_id信息
        self.param_fp16_list = []
        self.param_fp32_list = []
        self.momentum_fp32_list = []
        self.variance_fp32_list = []
        self.default_chunk_size = default_chunk_size

    def add_chunk(self, chunk_id, chunk_size, data_type, comm_group_id,
                  list_type: ChunkListType):
        """
        注册一个chunk信息
        @chunk_id: chunk的id
        @chunk_size: chunk尺寸
        @data_type: chunk中存储数据类型，由于PyTorch限制，再次说明不能在chunk里混合存储两种类型数据，或者将chunk内存以不同类型转换
        @comm_group_id: communication group id，历史原因名字不一致
        @list_type: chunk做在list的类型
        """
        self.dict_chunk_id_chunk_info[chunk_id] = (chunk_size, data_type)

        if comm_group_id not in self.comm_group_id_chunk_id_list:
            self.comm_group_id_chunk_id_list[comm_group_id] = list()
        self.comm_group_id_chunk_id_list[comm_group_id].append(chunk_id)
        self.dict_chunk_id_comm_group_id[chunk_id] = comm_group_id

        if list_type == ChunkListType.PARAM_FP16:
            self.param_fp16_list.append(chunk_id)
        elif list_type == ChunkListType.PARAM_FP32:
            self.param_fp32_list.append(chunk_id)
        elif list_type == ChunkListType.MOMENTUM:
            self.momentum_fp32_list.append(chunk_id)
        elif list_type == ChunkListType.VARIANCE:
            self.variance_fp32_list.append(chunk_id)
        else:
            raise RuntimeError

    def get_cur_chunk_num(self):
        return len(self.dict_chunk_id_chunk_info)

    def find_gap(self, numel, data_type):
        """
        在chunk list寻找满足numel大小，类型为data type的空隙
        TODO(jiaruifang) 应该优先分配同设备的gap
        实际使用场景非常具体：在fp16 BWD时，分配grad会在data的chunk内。
        """
        for chunk_id, tensor_info_list in self.dict_chunk_id_tensor_id.items():
            chunk_size, chunk_data_type = self.dict_chunk_id_chunk_info[
                chunk_id]
            if chunk_data_type != data_type or chunk_size < numel:
                continue
            prev_end = 0
            for tensor_id in tensor_info_list:
                info = self.dict_tensor_id_info[tensor_id]
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
        self.dict_tensor_id_info.clear()
        self.dict_chunk_id_tensor_id.clear()

    def generate_all_tensor_info(self):
        """
        展示每个chunk中tensor的状态
        """
        for chunk_id, tensor_info_list in self.dict_tensor_id_info.items():
            for tensor_id in tensor_info_list:
                yield self.dict_tensor_id_info[tensor_id]

    def generate_tensor_info_in_order(self, chunk_id):
        """
        产生在chunk id的所有tensor，以start_offset位置从小到大排序
        O(N)
        """
        for tensor_id in self.dict_chunk_id_tensor_id.get(chunk_id, []):
            yield self.dict_tensor_id_info[tensor_id]

    def get_tensor_info(self, tensor_id):
        return self.dict_tensor_id_info[tensor_id]

    def _binary_search(self, tensor_id_list, start_offset, start, end):
        # we need to distinugish whether we should insert
        # before or after the left boundary.
        # imagine [0] is the last step of the binary search
        # and we need to decide where to insert -1
        if start == end:
            if self.dict_tensor_id_info[
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
        mid_start_offset = self.dict_tensor_id_info[
            tensor_id_list[mid]].start_offset
        if mid_start_offset < start_offset:
            return self._binary_search(tensor_id_list, start_offset, mid + 1,
                                       end)
        elif mid_start_offset > start_offset:
            return self._binary_search(tensor_id_list, start_offset, start,
                                       mid - 1)
        else:
            return mid

    def add_tensor(self, chunk_id, tensor_id, start_offset, numel, param):
        """
        添加一个tensor，注册它所属的chunk_id和start_offset信息
        需要将chunk_id内的tensor按照start_offset排序
        二分查找时间复杂度O(logN)
        考虑chunk内排布的tensor可能不连续的情况
        """
        if chunk_id not in self.dict_chunk_id_tensor_id:
            self.dict_chunk_id_tensor_id[chunk_id] = list()

        tensor_id_list = self.dict_chunk_id_tensor_id[chunk_id]
        # 二分查找按照start_offset顺序从小到大插入
        pos = self._binary_search(tensor_id_list, start_offset, 0,
                                  len(tensor_id_list) - 1)
        tensor_id_list.insert(pos, tensor_id)
        if not is_param_registed(param):
            param_name = None
        else:
            param_name = param.ps_attr.name

        self.dict_tensor_id_info[tensor_id] = TensorInfo(
            chunk_id, tensor_id, start_offset, numel, param, param_name)

        if is_param_registed(param):
            param.ps_attr.data_chunk_id = chunk_id

    def delete_chunk_id(self, chunk_id):
        """
        @depracated，在静态chunk_schema中不应该被调用
        删除chunk_id对应chunk的索引信息
        """
        raise NotImplementedError
        # 删除chunk中的tensor
        if self.dict_chunk_id_tensor_id.get(chunk_id) is None:
            # logger.info(f'delete_chunk_id {chunk_id} does not exist')
            return
        for tid in self.dict_chunk_id_tensor_id.get(chunk_id, []):
            del self.dict_tensor_id_info[tid]

        del self.dict_chunk_id_tensor_id[chunk_id]

    def tensor_id_to_chunk_id(self, tensor_id) -> int:
        """
        tensor_id -> chunk_id
        """
        # info =
        info = self.dict_tensor_id_info.get(tensor_id)
        if info is None:
            return None
        else:
            return info.chunk_id

    def get_chunk_id(self, param: torch.nn.Parameter) -> int:
        tensor_id = param.ps_attr.id()
        info = self.dict_tensor_id_info.get(tensor_id)
        if info is None:
            return None
        return info.chunk_id

    def get_comm_group_id(self, chunk_id: int) -> List[int]:
        return self.comm_group_id_chunk_id_list[
            self.dict_chunk_id_comm_group_id[chunk_id]]

    def generate_all_chunks(self, chunk_list):
        for chunk_id, _ in self.dict_chunk_id_tensor_id.items():
            chunk = chunk_list[chunk_id]
            comm_group_id = self.dict_chunk_id_comm_group_id[chunk_id]
            yield chunk_id, comm_group_id, chunk

    def visit_chunk(self, chunk):
        rank = torch.distributed.get_rank()
        if rank != 1:
            return
        chunk_id = chunk.chunk_id
        comm_group_id = self.dict_chunk_id_comm_group_id[chunk_id]
        logger.info(
            f'rank {rank} Chunk id {chunk.chunk_id}, status {chunk.status()}, '
            f'global chunk id {comm_group_id}, capacity {chunk.capacity} elems, '
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
        total_bytes = 0
        logger.info(f'visit chunks')
        for chunk_id, _ in self.dict_chunk_id_tensor_id.items():
            chunk = chunk_list[chunk_id]
            comm_group_id = self.dict_chunk_id_comm_group_id[chunk_id]
            assert comm_group_id is not None

            logger.info(
                f'rank {rank} Chunk id {chunk.chunk_id}, status {chunk.status()}, '
                f'global chunk id {comm_group_id}, capacity {chunk.capacity} elems, '
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
        logger.info(f'OVERALL CHUNK SIZE {total_bytes/1e9} GB')

    def _get_tensor_id_list(self, chunk_id):
        if chunk_id not in self.dict_chunk_id_tensor_id:
            self.dict_chunk_id_tensor_id[chunk_id] = list()
        return self.dict_chunk_id_tensor_id[chunk_id]

    def try_insert_tensor(self, chunk_id, param) -> bool:
        """
        尝试向chunk内插入tensor，返回值表示是否成功
        """
        tensor_id_list = self._get_tensor_id_list(chunk_id)
        prev_end_pos = 0
        assert is_param_registed(param)
        numel = param.ps_attr.numel
        target_tensor_id = param.ps_attr.id()
        for idx, tensor_id in enumerate(tensor_id_list):
            tensor_info = self.dict_tensor_id_info[tensor_id]
            start_pos = tensor_info.start_offset
            gap = start_pos - prev_end_pos
            if gap >= numel:
                self.dict_tensor_id_info[target_tensor_id] = TensorInfo(
                    chunk_id, target_tensor_id, prev_end_pos, numel, param)
                tensor_id_list.insert(idx + 1, target_tensor_id)
                return True
            prev_end_pos = start_pos + tensor_info.numel

        logger.info(
            f'default_chunk_size {self.default_chunk_size}, prev_end_pos {prev_end_pos}, numel {numel}'
        )
        if self.default_chunk_size - prev_end_pos >= numel:
            self.dict_tensor_id_info[target_tensor_id] = TensorInfo(
                chunk_id, target_tensor_id, prev_end_pos, numel, param)
            tensor_id_list.insert(len(tensor_id_list), target_tensor_id)
            return True
        return False
