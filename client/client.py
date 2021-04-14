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
import os
from manager import HybridPSManager
from typing import Dict
import datetime
import logging
from torch.multiprocessing import Process, Manager

from utils import AccessType, PSChunkStatus, PSTensorStatus
from utils import TensorInfo, Chunk
from utils import ChunkList
from utils.helper import getsizeof


class HybridPSClient(object):
    def __init__(self,
                 gpu_index: int = 0,
                 data_type: torch.dtype = torch.float,
                 default_chunk_size: int = 64):
        """
        管理一个Process的Param, AccGrad, OS数据。
        每个进程可以访问一个GPU的显存，和cpu的内存
        功能:
          1. 充分利用cpu和gpu内存
          2. 细粒度调度，HybridPSClient包含若干chunk
        """
        self.pid = os.getpid()

        # index of gpu
        self.gpu_index = gpu_index
        self.data_type = data_type

        self.chunk_list = ChunkList(default_chunk_size)
        self.default_chunk_size = default_chunk_size

        self.module = None
        self.ps_id = -1
        self.param_data_dict = {}
        self.param_grad_dict = {}
        self.dict_tensor_id_chunk_id = {}

    def get_chunk_id(self, param: torch.nn.Parameter, access_type: AccessType):
        if access_type == AccessType.DATA:
            chunk_id = self.dict_tensor_id_chunk_id.get(param.ps_data_id)
        elif access_type == AccessType.GRAD:
            chunk_id = self.dict_tensor_id_chunk_id.get(param.ps_grad_id)
        else:
            raise TypeError("get_chunk_id access type {AccessType} is invalid")
        return chunk_id

    def prepare_device(self, target_device: torch.device, need_bytes: int):
        """
        让target device做好分配need_bytes大小空间的准备
        具体操作是找到
        TODO(jiaruifang)目前只考虑单GPU的情况
        """
        logging.log(
            logging.DEBUG,
            f'prepare_device target device {target_device} need size {need_bytes} bytes'
        )
        ps_manager = HybridPSManager()
        if ps_manager.max_mem(target_device.type,
                              target_device.index) < need_bytes:
            logging.log(
                logging.ERROR,
                f"{target_device} has not enough space for {need_bytes} elements"
            )
            raise RuntimeError

        extra_need_bytes = need_bytes - ps_manager.available_mem(
            target_device.type, target_device.index)
        # 不需要新分配
        if extra_need_bytes <= 0:
            return

        logging.log(
            logging.DEBUG,
            f'the device {target_device} has no enough free space, extra size is {extra_need_bytes}'
        )
        # 需要在target_device上腾出空间
        moved_list = self.chunk_list.chunk_to_move_out_for_room_making(
            extra_need_bytes, target_device)

        # TODO(jiaruifang)只考虑单卡情况，新设备只有gpu和cpu
        new_device = torch.device(
            'cpu') if target_device.type == 'cuda' else torch.device('cuda:0')
        logging.log(logging.DEBUG, f'moved list is {moved_list}')
        # 把他们移动到新设备上
        for idx in moved_list:
            self.chunk_move(idx, new_device)

    def access(self, param: torch.nn.Parameter, access_type: AccessType,
               compute_device: torch.device):
        """
        访问一个module中的param的data或者grad，将正确的数据加载
        找到param对应的chunk。
        1. 如果chunk存在
        然后决定是否移动chunk到本地设备，移动之前要给设备腾出足够空间。
        2. 如果chunk不存在
        比如grad FP16，在step过程所在的chunk已经被标记为FREE，并被释放掉。
        需要分配一个新的Chunk
        """
        if not self.is_ps_param(param):
            raise RuntimeError(
                "access a param not ps_data_tensor through HybridPS API")
        # tensor_id to chunk_id
        chunk_id = self.get_chunk_id(param, access_type)
        if chunk_id is None:
            # allocate a new tensor on compute device
            if access_type == AccessType.DATA:
                param.ps_data_tensor, param.ps_data_chunk_id = self.new_tensor(
                    param.shape, param.dtype, param.ps_data_id)
                current_device = param.ps_data_tensor.device
                self.param_grad_dict[param.ps_data_id] = param
                chunk_id = param.ps_data_chunk_id
            elif access_type == AccessType.GRAD:
                param.ps_grad_tensor, param.ps_grad_chunk_id = self.new_tensor(
                    param.shape, param.dtype, param.ps_grad_id)
                current_device = param.ps_grad_tensor.device
                self.param_grad_dict[param.ps_grad_id] = param
                chunk_id = param.ps_grad_chunk_id
            else:
                raise RuntimeError
            current_device = compute_device

        if access_type == AccessType.DATA:
            current_device = param.ps_data_tensor.device
        elif access_type == AccessType.GRAD:
            current_device = param.ps_grad_tensor.device
        else:
            raise RuntimeError

        if compute_device != current_device:
            self.prepare_device(
                compute_device, self.chunk_list[chunk_id].capacity *
                getsizeof(self.chunk_list[chunk_id].data_type))
            self.chunk_move(chunk_id, compute_device)

        self.chunk_list[chunk_id].touch()

        # 访问之后应该更新chunk tensor_infos的状态
        if access_type == AccessType.DATA:
            param.data = param.ps_data_tensor.data
            self.chunk_list[chunk_id].tensor_info_list.set_status(
                param.ps_data_id, PSTensorStatus.COMPUTE)
        elif access_type == AccessType.GRAD:
            # 在gpu上计算完毕，只改变grad很危险, master_p和model_p的data先传到cuda上
            #TODO(jiaruifang)权宜之计，只有在FP16_optimizer中，才出现让grad和data在不同设备的情况。
            if param.device != param.ps_grad_tensor.device:
                logging.warning(
                    f'param data is on {param.device}, while move grad to {param.ps_grad_tensor.device}'
                )
                param.data = torch.zeros(param.size(),
                                         dtype=param.dtype,
                                         device=param.ps_grad_tensor.device)
            assert param.device == param.ps_grad_tensor.device
            param.grad = param.ps_grad_tensor
            self.chunk_list[chunk_id].tensor_info_list.set_status(
                param.ps_grad_id, PSTensorStatus.COMPUTE)

    def access_data(self, param: torch.nn.Parameter,
                    compute_device: torch.device):
        self.access(param, AccessType.DATA, compute_device)

    def access_grad(self, param: torch.nn.Parameter,
                    compute_device: torch.device):
        self.access(param, AccessType.GRAD, compute_device)

    def release(self,
                param: torch.nn.Parameter,
                access_type: AccessType,
                reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        """
        这个param的data, grad不再需要放在计算设备，或者不需要hold
        TODO(jiaruifang)释放内存 or 只是不再计算设备的hold
        """
        chunk_id = self.get_chunk_id(param, access_type)
        if chunk_id is None:
            return
        if access_type == AccessType.DATA:
            self.chunk_list[chunk_id].tensor_info_list.set_status(
                param.ps_data_id, reset_to_status)
        elif access_type == AccessType.GRAD:
            self.chunk_list[chunk_id].tensor_info_list.set_status(
                param.ps_grad_id, reset_to_status)

    def release_data(self,
                     param: torch.nn.Parameter,
                     reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        """
        可以把一个tensor释放成FREE，也可以成HOLD
        """
        self.release(param, AccessType.DATA, reset_to_status)

    def release_grad(self,
                     param: torch.nn.Parameter,
                     reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        self.release(param, AccessType.GRAD, reset_to_status)

    def new_tensor(self, shape: torch.Size, data_type: torch.dtype,
                   tensor_id: int):
        """
        在PS上新分配shape大小空间, tensor_id是tensor在本进程内唯一标识
        TODO(jiaruifang) 现在的分配方式很简单，没考虑chunk空间可以释放的情况。
        只检查最后一个chunk是否有空余，如果没有分配新的chunk
        这个函数最后要注册tensor_id和chunk_id的对应关系，
        未来需要用tensor_id来索引chunk_id，chunk_id索引chunk
        chunk_list顺序递增
        """
        numel = 1
        for elem in shape:
            numel *= elem

        chunk_id, dest = self.chunk_list.allocate(numel, data_type, tensor_id)
        logging.log(
            logging.DEBUG,
            f'pid {self.pid}, allocates a tensor {shape} of {data_type} data on chunk {chunk_id}'
        )
        if tensor_id is not None:
            self.dict_tensor_id_chunk_id[tensor_id] = chunk_id
        return dest.view(shape), chunk_id

    @staticmethod
    def is_ps_param(parameter: torch.nn.Parameter):
        return hasattr(parameter, 'ps_data_id')

    def generate_id(self):
        self.ps_id = self.ps_id + 1
        return self.ps_id

    def is_ps_data(self, parameter: torch.nn.Parameter):
        return hasattr(parameter, 'ps_data_id')

    def is_ps_grad(self, parameter: torch.nn.Parameter):
        return hasattr(parameter, 'ps_grad_id')

    def _convert_to_ps_data(self, param: torch.nn.Parameter):
        if param.data is not None:
            # 初始化ps_data_tensor空间，并向其拷贝数据
            param.ps_data_tensor, param.ps_data_chunk_id = self.new_tensor(
                param.shape, param.dtype, param.ps_data_id)
            one_dim_param = param.data.contiguous().view(-1)
            param.ps_data_tensor.copy_(one_dim_param.view(param.ps_shape))
            param.data = param.ps_data_tensor.data
            self.param_data_dict[param.ps_data_id] = param

    def _convert_to_ps_grad(self, param: torch.nn.Parameter):
        # 初始化ps_grad_tensor空间，并向其拷贝数据
        if param.grad is not None:
            param.ps_grad_tensor, param.ps_gard_chunk_id = self.new_tensor(
                param.shape, param.dtype, param.ps_grad_id)
            one_dim_grad = param.grad.contiguous().view(-1)
            param.ps_grad_tensor.copy_(one_dim_grad.view(param.ps_shape))
            param.grad = param.ps_grad_tensor.data
            self.param_grad_dict[param.ps_grad_id] = param

    def _init_ps_param(self, param: torch.nn.Parameter):
        """
        在Parameter里面增加shape信息，生成id
        """
        if self.is_ps_param(param):
            logging.debug('param has already been a ps param')
            return

        param.ps_numel = param.numel()
        param.ps_shape = param.shape

        if not self.is_ps_data(param):
            param.ps_data_id = self.generate_id()
            param.ps_data_tensor = None

        if not self.is_ps_grad(param) and param.requires_grad is True:
            param.ps_grad_id = self.generate_id()
            param.ps_grad_tensor = None

    def register_module(self, module: torch.nn.Module):
        """
        将模型每个layer的param由HybridPS管理
        grad内存应该分配在一起
        data内存应该分配在一起
        """
        if module is not None:
            assert isinstance(module, torch.nn.Module)
            self.module = module

            # Note(jiaruifang) do we need recurse?
            for param in module.parameters(recurse=True):
                self._init_ps_param(param)

            # 如果有data和grad数据，要移动给HybridPS
            for param in module.parameters(recurse=True):
                self._convert_to_ps_data(param)

            for param in module.parameters(recurse=True):
                self._convert_to_ps_grad(param)
        self.release_all_grad()

    def register_param(self, src_param: torch.nn.Parameter):
        """
        @deprecated, used for debug
        Register a parameter to HybridPSClient's payload.
        Tensors (data, grad) in Param are flatten and concated in a contigous memory space.
        """
        self._init_ps_param(src_param)
        self._convert_to_ps_data(src_param)
        self._convert_to_ps_grad(src_param)

    def visit(self):
        for idx, chunk in self.chunk_list.generate():
            print(f"chunk {idx} on device {chunk.device} {chunk.get_status()}")
            chunk.visit()

    def chunk_move(self, chunk_id: int, device: torch.device):
        """
    将chunk_id的chunk移动到device上
    需要对对应param重新赋值
    """
        logging.debug(
            f'chunk_move chunk id {chunk_id} from {self.chunk_list[chunk_id].device} to {device}'
        )
        if self.chunk_list[chunk_id].device != device:
            logging.log(
                logging.DEBUG,
                f'pid {self.pid} move chunk {chunk_id} from {self.chunk_list[chunk_id].device} to {device}'
            )
            self.chunk_list[chunk_id].move(self.param_data_dict,
                                           self.param_grad_dict, device)

    def release_all_grad(self):
        if self.module is not None:
            for n, p in self.module.named_parameters():
                self.release_grad(p)

    def allreduce(self, local_tensor):
        """
    必须所有process同时执行，规约后的payload存储在哪(cpu or gpu)由调度器决定
    """
        pass

    def broadcast(self, local_tensor: torch.Tensor):
        """
    必须所有process同时执行，规约后的payload存储在哪由调度器决定
    """
        pass
