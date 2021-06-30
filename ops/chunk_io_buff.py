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

import torch
from client import ChunkList, ChunkTensorIndex
from client.parameter import is_torch_param
from manager import PatrickStarManager


class FP16ChunkWriteBuffer(object):
    def __init__(self, chunk_list: ChunkList,
                 chunk_tensor_index: ChunkTensorIndex, chunk_size: int,
                 compute_device: torch.device):
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index
        mgr = PatrickStarManager()
        mgr.add(compute_device.type, chunk_size * 4)
        self.payload = torch.zeros(chunk_size,
                                   dtype=torch.half,
                                   device=compute_device)
        self.cached_chunk_id = None

    def write_from_cache(self, param, data_tensor):
        """
        如果param是chunk中的最后一个tensor，则把data_tensor写到chunk中param对应的位置
        data_tensor
        """
        if is_torch_param(param):
            return param.data.copy_(data_tensor)
        else:
            info = self.chunk_tensor_index.get_tensor_info(
                param.ps_attr.data_id())

            if self.cached_chunk_id is not None and info.chunk_id != self.cached_chunk_id:
                # TODO CPU->GPU拷贝需要优化
                print(f'write chunk {self.cached_chunk_id}')
                self.chunk_list[self.cached_chunk_id].payload.copy_(
                    self.payload)

            # print(f'data_tensor {data_tensor.shape} {data_tensor.view(-1).shape}')
            self.payload.narrow(0, info.start_offset,
                                info.numel).copy_(data_tensor.view(-1))
            self.cached_chunk_id = info.chunk_id

    def write_cached_chunk(self):
        """
        将cache住的payload写到chunk里
        """
        print(f'finally, write chunk {self.cached_chunk_id}')
        self.chunk_list[self.cached_chunk_id].payload.copy_(self.payload)
        self.cached_chunk_id = None


class FP32ChunkReadBuffer(object):
    """
    FP32 Chunk Buff用于加速ADAM计算时的数据传输。
    """
    def __init__(self, chunk_list: ChunkList,
                 chunk_tensor_index: ChunkTensorIndex, chunk_size: int,
                 compute_device: torch.device):
        """
        在compute_device分配一个FP32 Chunk作为缓存
        """
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index
        mgr = PatrickStarManager()
        mgr.add(compute_device.type, chunk_size * 4)
        self.payload = torch.zeros(chunk_size,
                                   dtype=torch.float,
                                   device=compute_device)
        self.cached_chunk_id = None

    def access_from_cache(self, param) -> torch.Tensor:
        """
        访问param，如果param是chunk的第一个tensor则触发chunk拷贝
        返回一个tensor内存
        """
        if is_torch_param(param):
            return param.data
        else:
            info = self.chunk_tensor_index.get_tensor_info(
                param.ps_attr.data_id())
            if info.start_offset == 0:
                # TODO CPU->GPU拷贝需要优化
                self.payload.copy_(self.chunk_list[info.chunk_id].payload)
                self.cached_chunk_id = info.chunk_id
            else:
                assert info.chunk_id == self.cached_chunk_id
            return self.payload.narrow(0, info.start_offset, info.numel)
