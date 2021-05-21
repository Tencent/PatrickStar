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
import logging
import torch
from .const import AccessType, PSChunkStatus, PSTensorStatus
from .chunk_list import ChunkList
import utils.global_timer as global_timer
import time
from .parameter import PSParameter


class TensorInfo(object):
    """
    记录chunk内存存储tensor的属性,
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

    def status(self):
        """
        访问param中的成员变量很慢
        """
        return self.param.ps_attr.get_status(self.access_type)

    def showme(self):
        logging.info(
            f'tensor_id {self.tensor_id}, name {self.tensor_name}, shape {self.param.ps_shape}, chunk_id {self.chunk_id}, start_offset {self.start_offset}, nueml {self.numel}, status {self.status()}'
        )


class ChunkTensorIndex(object):
    def __init__(self):
        """
        存储parameter和chunk的检索信息
        """
        # 1-1 dict, tensor_id -> TensorInfo
        self.dict_tensor_id_info: dict[int, TensorInfo] = {}
        # 1-N dict, chunk_id -> List(tensor_id) in order of start_offset
        self.dict_chunk_id_tensor_id: dict[int, List[int]] = {}
        self.dict_chunk_id_chunk_info: dict[int, tuple] = {}

    def add_chunk(self, chunk_id, chunk_size, data_type):
        self.dict_chunk_id_chunk_info[chunk_id] = (chunk_size, data_type)

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

    def generate_grad_tensor_param(self):
        """
        按chunk内部排列顺序生成所有当前没有被free的grad tensor所在的param
        """
        res_list = []
        for chunk_id, tensor_id_list in self.dict_chunk_id_tensor_id.items():
            for tensor_id in tensor_id_list:
                info = self.dict_tensor_id_info[tensor_id]
                if info.access_type == AccessType.GRAD and info.status(
                ) != PSTensorStatus.FREE:
                    res_list.append(info.param)
        return res_list

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

    def add_tensor(self, chunk_id, tensor_id, start_offset, numel, param,
                   access_type):
        """
        添加一个tensor，注册它所属的chunk_id和start_offset信息
        需要将chunk_id内的tensor按照start_offset排序
        二分查找时间复杂度O(logN)
        """
        if chunk_id not in self.dict_chunk_id_tensor_id:
            self.dict_chunk_id_tensor_id[chunk_id] = list()

        tensor_id_list = self.dict_chunk_id_tensor_id[chunk_id]
        # 二分查找按照start_offset顺序从小到大插入
        pos = self._binary_search(tensor_id_list, start_offset, 0,
                                  len(tensor_id_list) - 1)
        tensor_id_list.insert(pos, tensor_id)
        if not hasattr(param, 'ps_attr'):
            param_name = None
        else:
            param_name = param.ps_attr.ps_name
        self.dict_tensor_id_info[tensor_id] = TensorInfo(
            chunk_id, tensor_id, start_offset, numel, param, access_type,
            param_name)

        if access_type == AccessType.DATA:
            param.ps_attr.ps_data_chunk_id = chunk_id
        elif access_type == AccessType.GRAD:
            param.ps_attr.ps_grad_chunk_id = chunk_id

    def delete_chunk_id(self, chunk_id):
        """
        @depracated，在静态chunk_schema中不应该被调用
        删除chunk_id对应chunk的索引信息
        """
        raise NotImplementedError
        # 删除chunk中的tensor
        if self.dict_chunk_id_tensor_id.get(chunk_id) is None:
            # logging.info(f'delete_chunk_id {chunk_id} does not exist')
            return
        for tid in self.dict_chunk_id_tensor_id.get(chunk_id, []):
            del self.dict_tensor_id_info[tid]

        del self.dict_chunk_id_tensor_id[chunk_id]

    def delete_tensor(self, tensor_id):
        """
        删除一个tensor，可能导致一个chunk也被删除，应对内存进行释放
        """
        raise NotImplementedError
        cid_delete_list = []
        for cid, tid_list in self.dict_chunk_id_tensor_id.items():
            if tensor_id in tid_list:
                tid_list.remove(tensor_id)

            if len(tid_list) == 0:
                cid_delete_list.append(cid)

        for cid in cid_delete_list:
            del self.dict_chunk_id_tensor_id[cid]

        del self.dict_tensor_id_info[tensor_id]

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

    def get_chunk_id(self, param: torch.nn.Parameter,
                     access_type: AccessType) -> int:
        tensor_id = param.ps_attr.get_tensor_id(access_type)
        info = self.dict_tensor_id_info.get(tensor_id)
        if info is None:
            return None
        return info.chunk_id

    def visit_chunks(self, chunk_list: ChunkList):
        total_bytes = 0
        for chunk_id, _ in self.dict_chunk_id_tensor_id.items():
            chunk = chunk_list[chunk_id]
            logging.info(
                f'Chunk id {chunk.chunk_id}, capacity {chunk.capacity} dtype {chunk.data_type}, size {chunk.get_size()}, device {chunk.get_device()}'
            )
            for info in self.generate_tensor_info_in_order(chunk_id):
                assert info.chunk_id == chunk_id, f'{info.chunk_id} vs {chunk_id}'
                logging.info(
                    f"** tensor: chunk_id {chunk_id}, start {info.start_offset}, end {info.start_offset + info.numel}, size {info.numel}, tensor_id {info.tensor_id}, status {info.status()}, name {info.tensor_name}"
                )
            total_bytes += chunk.get_size()
        logging.info(f'OVERALL CHUNK SIZE {total_bytes/1e9} GB')
