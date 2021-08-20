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
from patrickstar.core import ChunkList, ChunkTensorIndex
from patrickstar.core.parameter import is_torch_param
from patrickstar.manager import PatrickStarManager
from patrickstar.utils import logger
from patrickstar.deepspeed_helper.global_vars import get_args


class FP16ChunkWriteBuffer(object):
    def __init__(self, chunk_list: ChunkList,
                 chunk_tensor_index: ChunkTensorIndex, chunk_size: int):
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index
        self.cached_src_chunk_id = None
        self.cached_target_chunk_id = None
        args = get_args()
        if args.use_gpu_fp32_convert_for_adam:
            self.gpu_fp16_buff = torch.zeros(
                chunk_size,
                dtype=torch.half,
                device=torch.device(f'cuda:{args.local_rank}'))

    def write_from_cache(self, target_param, src_param):
        """
        如果src_param是chunk中的最后一个tensor，
        则把src_param的chunk写到target_param所在的chunk中
        可能是cpu向gpu移动
        """
        args = get_args()
        if is_torch_param(src_param):
            return target_param.data.copy_(src_param.data)
        else:
            src_info = self.chunk_tensor_index.get_tensor_info(
                src_param.ps_attr.id())
            target_info = self.chunk_tensor_index.get_tensor_info(
                target_param.ps_attr.id())

            if self.cached_src_chunk_id is not None and src_info.chunk_id != self.cached_src_chunk_id:
                # TODO CPU->GPU拷贝需要优化
                target_device = self.chunk_list[
                    self.cached_target_chunk_id].payload.device
                src_device = self.chunk_list[
                    self.cached_src_chunk_id].payload.device
                logger.info(
                    f'Write chunk {self.cached_src_chunk_id} -> {self.cached_target_chunk_id}, '
                    f'{src_device} -> {target_device}'
                )
                if args.use_gpu_fp32_convert_for_adam and target_device.type == 'cuda' and src_device.type == 'cpu':
                    self.gpu_fp16_buff.copy_(
                        self.chunk_list[self.cached_src_chunk_id].payload)
                    self.chunk_list[self.cached_target_chunk_id].payload.copy_(
                        self.gpu_fp16_buff)
                else:
                    self.chunk_list[self.cached_target_chunk_id].payload.copy_(
                        self.chunk_list[self.cached_src_chunk_id].payload)
            self.cached_src_chunk_id = src_info.chunk_id
            self.cached_target_chunk_id = target_info.chunk_id

    def reset(self):
        """
        reset时，将cache住的payload写到chunk里
        """
        args = get_args()
        logger.info(
            f'rank {args.local_rank} finally, write chunk {self.cached_target_chunk_id}'
        )
        # Note 有可能这一个进程只有一个Chunk，且该Chunk只有一个Embedding Layer，而它是Torch管理的
        # 导致这个Chunk没有分配payload
        if self.chunk_list[self.cached_src_chunk_id] is not None:
            self.chunk_list[self.cached_target_chunk_id].payload.copy_(
                self.chunk_list[self.cached_src_chunk_id].payload)
        self.cached_src_chunk_id = None
        self.cached_target_chunk_id = None


class FP32ChunkReadBuffer(object):
    """
    FP32 Chunk Buff用于加速ADAM计算时的数据传输。
    可能读入CPU或者GPU之中
    """
    def __init__(self, chunk_list: ChunkList,
                 chunk_tensor_index: ChunkTensorIndex, chunk_size: int,
                 margin_chunk_num_for_gpu_adam: int):
        """
        在compute_device分配一个FP32 Chunk作为缓存
        @params
        margin_chunk_num_for_gpu_adam: 训练过程GPU可以额外分配给adam的chunk个数
        """
        args = get_args()
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index
        self.cpu_payload = torch.empty(chunk_size,
                                       dtype=torch.float,
                                       device=torch.device('cpu:0'),
                                       pin_memory=True)
        self.fp32_gpu_buffer = torch.empty(
            chunk_size,
            dtype=torch.float,
            device=torch.device(f'cuda:{args.local_rank}'))
        logger.info(
            f"Allocate fp32 Chunk Buffer of size {chunk_size/1e6} MB on CPU.")
        if margin_chunk_num_for_gpu_adam > 0:
            if args.use_fake_dist:
                gpu_device = torch.device(f'cuda:0')
            else:
                gpu_device = torch.device(f'cuda:{args.local_rank}')
            self.gpu_payload = torch.empty(chunk_size,
                                           dtype=torch.float,
                                           device=gpu_device)
            logger.info(
                f"Allocate fp32 Chunk Buffer of size {chunk_size/1e6} MB on {gpu_device}."
            )
        self.cached_chunk_id = None
        self.margin_chunk_num_for_gpu_adam = margin_chunk_num_for_gpu_adam
        self.cached_chunk_num = 0
        self.ret_payload = None

    def get_cached_chunk_num(self):
        """
        获得被cache过的chunk数目
        """
        return self.cached_chunk_num

    def access_from_cache(self, param) -> torch.Tensor:
        """
        访问param，如果param是chunk的第一个tensor则触发chunk拷贝
        target_device可能是cpu或者gpu，因此需要两种不同的payload来缓存
        返回一个tensor内存
        """
        args = get_args()
        if is_torch_param(param):
            return param.data
        else:
            info = self.chunk_tensor_index.get_tensor_info(
                param.ps_attr.id())

            # 触发cached chunk更新，判断条件是param.data Tensor是Chunk的第一个Tensor
            if info.start_offset == 0:
                # TODO GPU FP16->CPU FP32拷贝需要优化,
                self.cached_chunk_num += 1

                if self.get_cached_chunk_num(
                ) < self.margin_chunk_num_for_gpu_adam:
                    target_device = torch.device(f'cuda:{args.local_rank}')
                else:
                    target_device = torch.device('cpu:0')

                if target_device.type == 'cuda':
                    self.gpu_payload.copy_(
                        self.chunk_list[info.chunk_id].payload)
                    self.ret_payload = self.gpu_payload
                elif target_device.type == 'cpu':
                    if args.use_gpu_fp32_convert_for_adam:
                        self.fp32_gpu_buffer.copy_(
                            self.chunk_list[info.chunk_id].payload)
                        self.cpu_payload.copy_(self.fp32_gpu_buffer)
                    else:
                        self.cpu_payload.copy_(
                            self.chunk_list[info.chunk_id].payload)
                    self.ret_payload = self.cpu_payload
                logger.info(
                    f'read chunk to cache {self.cached_chunk_id} {self.chunk_list[info.chunk_id].payload.device} '
                    f'({self.chunk_list[info.chunk_id].payload.dtype}) -> {self.ret_payload.device} (Float)'
                )
                self.cached_chunk_id = info.chunk_id
            else:
                assert info.chunk_id == self.cached_chunk_id
            return self.ret_payload.narrow(0, info.start_offset, info.numel)

    def reset(self):
        self.cached_chunk_num = 0
        self.ret_payload = None
        self.cached_chunk_id = None
