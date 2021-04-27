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

from .const import AccessType, PSChunkStatus, PSTensorStatus
from .chunk_data import Chunk
from .chunk_list import ChunkList
from .helper import getsizeof
from .chunk_tensor_index import ChunkTensorIndex


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

        self.chunk_tensor_index = ChunkTensorIndex()

    def get_chunk_id(self, param: torch.nn.Parameter, access_type: AccessType):
        """
        Get chunk id of a tensor (data, grad) of a Parameter.
        return None indicates the tensor has not been allocated to any chunk.
        """
        if access_type == AccessType.DATA:
            chunk_id = self.chunk_tensor_index.tensor_id_to_chunk_id(
                param.ps_data_id)
        elif access_type == AccessType.GRAD:
            chunk_id = self.chunk_tensor_index.tensor_id_to_chunk_id(
                param.ps_grad_id)
        else:
            raise TypeError("get_chunk_id access type {AccessType} is invalid")
        return chunk_id

    def prepare_device(self, target_device: torch.device, need_bytes: int):
        """
        让target device做分配need_bytes大小空间的准备
        如果空间不足，需要在目标设备上释放或者移动出一些chunk。
        """
        logging.log(
            logging.DEBUG,
            f'prepare_device target device {target_device} need size {need_bytes} bytes'
        )
        ps_manager = HybridPSManager()
        max_mem = ps_manager.max_mem(target_device.type, target_device.index)
        if max_mem < need_bytes:
            logging.log(
                logging.ERROR,
                f"{target_device} has not enough space for {need_bytes} elements"
            )
            # TODO(jiaruifang)可以爆表时候再释放
            raise RuntimeError

        available_size = ps_manager.available_mem(target_device.type,
                                                  target_device.index)
        extra_need_bytes = need_bytes - available_size

        logging.debug(
            f'{target_device} (max size {max_mem} B) now available size {available_size} B needs {need_bytes} B'
        )
        # 不需要新分配
        if extra_need_bytes <= 0:
            return

        logging.log(
            logging.DEBUG,
            f'the device {target_device} has no enough free space, extra size is {extra_need_bytes} bytes'
        )
        # 需要在target_device上腾出空间
        moved_list = self.chunk_list.chunk_to_move_out_for_room_making(
            extra_need_bytes, target_device, self.chunk_tensor_index)

        # TODO(jiaruifang)只考虑单卡情况，新设备只有gpu和cpu
        new_device = torch.device(
            'cpu') if target_device.type == 'cuda' else torch.device('cuda:0')

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
            if access_type == AccessType.DATA:
                self.new_data(param, self.chunk_tensor_index)
                current_device = param.ps_data_tensor.device
                chunk_id = param.ps_data_chunk_id
            elif access_type == AccessType.GRAD:
                self.new_grad(param, self.chunk_tensor_index)
                current_device = param.ps_grad_tensor.device
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
            #访问一个free状态的chunk，上面不会分配，此处把它释放了。
            self.prepare_device(compute_device,
                                self.chunk_list[chunk_id].get_size())
            self.chunk_move(chunk_id, compute_device)

        self.chunk_list[chunk_id].touch()

        # 访问之后应该更新chunk tensor_infos的状态
        if access_type == AccessType.DATA:
            param.data_status = PSTensorStatus.COMPUTE
            param.data = param.ps_data_tensor
        elif access_type == AccessType.GRAD:
            param.grad_status = PSTensorStatus.COMPUTE

    def access_data(self, param: torch.nn.Parameter,
                    compute_device: torch.device):
        """
        将param的ps_data_tensor的数据放置到compute_device上
        """
        self.access(param, AccessType.DATA, compute_device)

    def access_grad(self, param: torch.nn.Parameter,
                    compute_device: torch.device):
        """
        将param的ps_grad_tensor的数据放置到compute_device上
        NOTE，并没有正确设置param的grad，此时grad的数据无效。因为grad的设备属性并不自由，需要看data的脸色行事。我们使用grad时候，需要显式设置
        `param.grad = param.ps_grad_tensore`
        """
        self.access(param, AccessType.GRAD, compute_device)

    def release(self,
                param: torch.nn.Parameter,
                access_type: AccessType,
                reset_to_status: PSTensorStatus = PSTensorStatus.HOLD):
        """
        这个param的data, grad不再需要放在计算设备，或者不需要hold
        """
        chunk_id = self.get_chunk_id(param, access_type)
        logging.debug(
            f'release {access_type} chunk_id {chunk_id} to {reset_to_status}')
        if chunk_id is None:
            return

        # 设置data[grad]_status已经在chunk中删除这个tensor
        # chunk_tensor_index删除tensor索引，顺便判断是否有chunk需要删除
        # chunk list判断是否有chunk需要删除
        if access_type == AccessType.DATA:
            # 把data的内存删除，方式是将它指向一段长度为1的内存
            param.data = torch.zeros(1, dtype=param.dtype, device=param.device)
            param.data_status = reset_to_status
            if reset_to_status == PSTensorStatus.FREE:
                self.chunk_tensor_index.delete_tensor(param.ps_data_id)
        elif access_type == AccessType.GRAD:
            param.grad = None
            param.grad_status = reset_to_status
            if reset_to_status == PSTensorStatus.FREE:
                self.chunk_tensor_index.delete_tensor(param.ps_grad_id)
        #在这里立刻释放，被标记为free的chunks
        # TODO(jiaruifang)要记得释放每个tensor指向的内存，否则并么有真正释放内存
        self.chunk_list.delete_free_chunks(self.chunk_tensor_index)

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

    def new_data(self, param: torch.nn.Parameter,
                 chunk_tensor_index: ChunkTensorIndex):
        """
        为param分配data的ps tensor
        """
        chunk_id, dest = self.chunk_list.allocate(param, AccessType.DATA,
                                                  chunk_tensor_index)
        param.ps_data_tensor = dest.view(param.ps_shape)
        param.ps_data_chunk_id = chunk_id
        param.data_status = PSChunkStatus.HOLD

    def new_grad(self, param: torch.nn.Parameter,
                 chunk_tensor_index: ChunkTensorIndex):
        """
        为param分配grad的ps tensor
        """
        chunk_id, dest = self.chunk_list.allocate(param, AccessType.GRAD,
                                                  chunk_tensor_index)
        param.ps_grad_tensor = dest.view(param.ps_shape)
        param.ps_grad_chunk_id = chunk_id
        param.grad_status = PSChunkStatus.HOLD

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
            self.new_data(param, self.chunk_tensor_index)
            one_dim_param = param.data.contiguous().view(-1)
            param.ps_data_tensor.copy_(one_dim_param.view(param.ps_shape))
            param.data = param.ps_data_tensor
            param.data_status = PSTensorStatus.HOLD

    def _convert_to_ps_grad(self, param: torch.nn.Parameter):
        # 初始化ps_grad_tensor空间，并向其拷贝数据
        if param.grad is not None:
            self.new_grad(param, self.chunk_tensor_index)
            one_dim_grad = param.grad.contiguous().view(-1)
            param.ps_grad_tensor.copy_(one_dim_grad.view(param.ps_shape))
            param.grad = param.ps_grad_tensor
            param.grad_status = PSTensorStatus.HOLD

    def _init_ps_param(self, param: torch.nn.Parameter):
        """
        在Parameter里面增加shape信息，生成id
        """
        if self.is_ps_param(param):
            logging.debug('param has already been a ps param')
            return

        param.ps_numel = param.numel()
        param.ps_shape = param.shape
        param.data_status = PSTensorStatus.FREE
        param.grad_status = PSTensorStatus.FREE

        if not self.is_ps_data(param):
            param.ps_data_id = self.generate_id()
            # self.param_data_dict[param.ps_data_id] = param
            # self.chunk_tensor_index.register_tensor(param.ps_data_id, AccessType.DATA, param)
            param.ps_data_tensor = None

        if not self.is_ps_grad(param) and param.requires_grad is True:
            param.ps_grad_id = self.generate_id()
            # self.param_grad_dict[param.ps_grad_id] = param
            # self.chunk_tensor_index.register_tensor(param.ps_grad_id, AccessType.GRAD, param)
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
            logging.info(
                f"chunk {idx} on device {chunk.device} status {self.chunk_tensor_index.chunk_status(idx)}"
            )
            chunk.visit(self.chunk_tensor_index)

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
            self.chunk_list[chunk_id].move(self.chunk_tensor_index, device)

    def release_all_grad(self):
        if self.module is not None:
            for n, p in self.module.named_parameters():
                self.release_grad(p, PSTensorStatus.FREE)

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
