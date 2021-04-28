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
# from .chunk_data import Chunk
import utils.global_timer as global_timer
import time


class TensorInfo(object):
    """
    记录chunk内存存储tensor的属性,
    """
    def __init__(self, chunk_id: int, tensor_id: int, start_offset: int,
                 numel: int, param: torch.nn.Parameter,
                 access_type: AccessType):
        self.tensor_id = tensor_id
        self.chunk_id = chunk_id
        self.start_offset = start_offset
        self.numel = numel
        self.param = param
        self.access_type = access_type

    def status(self):
        """
        访问param中的成员变量很慢
        """
        if self.access_type == AccessType.DATA:
            if not hasattr(self.param, 'data_status'):
                return PSTensorStatus.UNINIT
            ret = self.param.data_status
        elif self.access_type == AccessType.GRAD:
            if not hasattr(self.param, 'grad_status'):
                return PSTensorStatus.UNINIT
            ret = self.param.grad_status
        return ret

    def showme(self):
        logging.info(
            f'tensor_id {self.tensor_id}, chunk_id {self.chunk_id}, start_offset {self.start_offset}, nueml {self.numel}, status {self.status()}'
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
        # 1-1 chunk_id -> Chunk
        self.dict_chunk_id_chunk: dict[int, Chunk] = {}

    def show_status(self):
        """
        展示每个chunk中tensor的状态
        """
        for chunk_id, tensor_info_list in self.dict_chunk_id_tensor_id.items():
            for tensor_id in tensor_info_list:
                yield self.dict_tensor_id_info[tensor_id]

    def generate_tensor_info_in_order(self, chunk_id):
        """
        产生在chunk id的所有tensor，以start_offset位置从小到大排序
        O(N)
        """
        for tensor_id in self.dict_chunk_id_tensor_id.get(chunk_id, []):
            yield self.dict_tensor_id_info[tensor_id]

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
        self.dict_tensor_id_info[tensor_id] = TensorInfo(
            chunk_id, tensor_id, start_offset, numel, param, access_type)

    def delete_chunk_id(self, chunk_id):
        """
        删除chunk_id对应chunk的索引信息
        """
        # 删除chunk中的tensor
        if self.dict_chunk_id_tensor_id.get(chunk_id) is None:
            # logging.info(f'delete_chunk_id {chunk_id} does not exist')
            return
        for tid in self.dict_chunk_id_tensor_id.get(chunk_id, []):
            del self.dict_tensor_id_info[tid]

        del self.dict_chunk_id_tensor_id[chunk_id]

    def delete_tensor(self, tensor_id):
        """
        删除一个tensor，可能导致一个chunk也被产出，应对内存进行释放
        """
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

    def chunk_status(self, chunk_id) -> PSChunkStatus:
        """
        chunk的状态，由它管理的tensor共同决定
        TODO(jiaruifang)速度很慢，有待优化
        """
        # if len(self.dict_chunk_id_tensor_id[chunk_id]) == 0:
        #     return PSChunkStatus.FREE

        free_flag = True
        # O (logK*V)
        for tensor_id in self.dict_chunk_id_tensor_id.get(chunk_id, []):
            # start_time = time.time()
            info = self.dict_tensor_id_info[tensor_id]

            start_time = time.time()
            # inline status()
            # 2.575 sec -> 1.676 s
            # no access to param.data_status/grad_status
            # 1.676 s  -> 0.94sec
            # if-else in one line
            # 0.94sec -> 0.761sec
            access_type = info.access_type
            status = info.param.data_status if (
                access_type == AccessType.DATA) else info.param.grad_status

            if status == PSTensorStatus.COMPUTE:
                global_timer.delete_free_chunks_part1 += time.time(
                ) - start_time
                return PSChunkStatus.COMPUTE
            elif status != PSTensorStatus.FREE:
                free_flag = False
            global_timer.delete_free_chunks_part1 += time.time() - start_time

        if free_flag is True:
            return PSChunkStatus.FREE
        else:
            return PSChunkStatus.HOLD
