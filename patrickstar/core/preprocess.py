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

import os
import torch
import functools

from patrickstar.utils import init_distributed, see_memory_usage
from patrickstar.utils import logger, print_rank
from patrickstar.core import PatrickStarClient, AccessType, ChunkListType, ChunkTensorIndex, ChunkList
from patrickstar.core import PSParameter, register_param, is_param_registed, register_torch_param
from patrickstar.core.chunk_schema_scheduler import ChunkCreator
from patrickstar.deepspeed_helper.global_vars import get_args
from typing import List
_orig_torch_empty = torch.empty


def empty_cpu_tensor_half(*size, **kwargs):
    if not 'device' in kwargs.keys():
        kwargs['device'] = torch.device('cpu:0')
    tensor = _orig_torch_empty(*size, **kwargs)
    if tensor.is_floating_point():
        return tensor.half()
    else:
        return tensor


def new_cpu_tensor_half(cls, *args):
    device = torch.device('cpu:0')
    tensor = torch.ones((1, 1), device=device).new_empty(*args).half()
    print_rank(
        f'During model initialization, a new tensor of shape {tensor.shape} is created.'
    )
    if tensor.is_floating_point():
        return tensor.half()
    else:
        return tensor


def empty_cpu_tensor(*size, **kwargs):
    if not 'device' in kwargs.keys():
        kwargs['device'] = torch.device('cpu:0')
    tensor = _orig_torch_empty(*size, **kwargs)
    return tensor


def new_cpu_tensor(cls, *args):
    device = torch.device('cpu:0')
    tensor = torch.ones((1, 1), device=device).new_empty(*args)
    return tensor


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self, config=None, dtype=None):
        self._set_dtype(config, dtype)
        assert self.dtype in [
            torch.half, torch.float
        ], f"Invalid data type {self.dtype}, allowed values are [torch.half, torch.float]"

    def __enter__(self):
        def preprocess_after(f):
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):
                print_rank(f'Before initializing {module.__class__.__name__}',
                           force=False)
                f(module, *args, **kwargs)
                self._post_init_method(module)
                print_rank(
                    f'After initializing followed by post init for {module.__class__.__name__}',
                    force=False)

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = preprocess_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = preprocess_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _enable_class(subclass)

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = torch.nn.modules.module.Module.__init_subclass__
        torch.Tensor.__old_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(
            _init_subclass)
        if self.dtype == torch.half:
            torch.Tensor.__new__ = new_cpu_tensor_half
            torch.empty = empty_cpu_tensor_half
        else:
            torch.Tensor.__new__ = new_cpu_tensor
            torch.empty = empty_cpu_tensor

    def __exit__(self, exc_type, exc_value, traceback):
        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = torch.nn.modules.module.Module._old_init_subclass

        torch.Tensor.__new__ = torch.Tensor.__old_new__
        torch.empty = _orig_torch_empty

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _set_dtype(self, ds_config, dtype):
        if dtype is None:
            self.dtype = torch.half
        else:
            self.dtype = dtype


class ChunkCreator(object):
    def __init__(self, default_chunk_size: int, chunk_list: ChunkList,
                 chunk_tensor_index: ChunkTensorIndex,
                 dummy_param_list: List[torch.nn.Parameter]):
        """
        chunk的构造器，通过append tensor隐式构建chunk和chunk tensor的映射关系
        """
        # default chunk size是subchunk size
        self.default_chunk_size = default_chunk_size
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index

        self.chunk_id = 0
        # accumulated elements count
        self.acc_cnt = 0
        # the cached data type
        self.data_type = None

        self.world_size = 1
        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()

        # fp16 tensors, fp32 tensors, m tensors, v tensors各组成一个list
        # list_id表示tensor在list的顺序。global id跨list需要清零
        self.list_id = 0
        self.comm_group_idx = 0

        self.dummy_param_list = dummy_param_list
        logger.info(f'default chunk size is {default_chunk_size}')

    def append_tensor(self, tensor_id, numel, param, access_type: AccessType,
                      data_type, chunk_list_type: ChunkListType) -> int:
        """
        向chunk_tensor_index注册tensor，当新的Tensor大小超过Chunk剩余空间，
        则开辟一个新的Chunk添加到ChunkList中，将Tensor分配在新Chunk的起始位置
        返回chunk在chunk group中的位置
        @tensor_id: tensor id
        @numel: tensor number of elements
        @param: an instance of torch.Paramater
        @access_type: define accessing which tensor of the param, either data or grad.
        @data_type: appended tenor type
        @chunk_list_type: the list type which the tensor appends to
        @is_copy, whether we copy the payload of param tensor to chunk
        """
        # data_type甚至可以和param dtype不一致
        self.data_type = data_type

    def start_new_chunk_list(self, add_dummy_chunk_flag: bool,
                             chunk_list_type: ChunkListType):
        """
        对构造中的chunk进行收尾
        """
        if self.acc_cnt > 0:
            self.chunk_list.new_chunk(self.chunk_id, self.default_chunk_size,
                                      self.data_type)
            self.chunk_tensor_index.add_chunk(self.chunk_id,
                                              self.default_chunk_size,
                                              self.data_type,
                                              self.comm_group_idx,
                                              chunk_list_type)
            self.chunk_id += 1
            self.acc_cnt = 0
            # 下一个chunk的list_id
            self.list_id += 1

        # 给不足world_size的global chunk补上dummy chunk，每个dummy chunk管理一个dummy param
        if add_dummy_chunk_flag:
            while self.list_id % self.world_size != 0:
                logger.info('add dummy chunk')
                self.chunk_list.new_chunk(self.chunk_id,
                                          self.default_chunk_size,
                                          self.data_type,
                                          is_dummy=True)
                self.chunk_tensor_index.add_chunk(self.chunk_id,
                                                  self.default_chunk_size,
                                                  self.data_type,
                                                  self.comm_group_idx,
                                                  chunk_list_type)
                self.dummy_param_list.append(
                    torch.nn.Parameter(torch.zeros(1, dtype=self.data_type),
                                       requires_grad=False))
                # 加入一个dummy param可以让dummy chunk状态被设置为hold
                register_param(self.dummy_param_list[-1], "dummy")
                self.chunk_tensor_index.add_tensor(
                    self.chunk_id, self.dummy_param_list[-1].ps_attr.data_id(),
                    0, 1, self.dummy_param_list[-1], AccessType.DATA)

                self.chunk_id += 1
                self.list_id += 1

        self.list_id = 0
        self.comm_group_idx += 1


class PSPreProcessCtx(InsertPostInitMethodToModuleSubClasses):
    """
    A context to initialize model
    """
    def __init__(self, client: PatrickStarClient, dtype=None):
        super().__init__(config=None, dtype=dtype)
        if not torch.distributed.is_initialized():
            assert torch.distributed.is_initialized(
            ), "Parameters cannot be scattered without initializing torch.distributed"
        args = get_args()
        self.rank = args.local_rank
        self.world_size = torch.distributed.get_world_size()
        self.client = client
        self.dummy_param_list = []

    def _post_init_method(self, module):
        """
        在model的param被PyTorch初始化完毕后完成
        1. 保留local的tensor，通过删除remote tensor的方式
        2. 将model param拷贝到chunk对应的内存中
        """
        see_memory_usage(
            f"Before converting parmas in {module.__class__.__name__}",
            force=False)
        args = get_args()
        if args.use_cpu_embedding:
            # cpu_embedding优化把embedding交给Torch管理而非Chunk
            if module.__class__.__name__ == 'Embedding':
                for name, param in module.named_parameters(recurse=False):
                    register_torch_param(param, f'embedding_{name}')
                return

        print_rank(f'Converting Params in {module.__class__.__name__}',
                   force=False)
        rank = args.local_rank

        # 在模型初始化的过程构造模型，post_init_method调用粒度是一个SubModule，比如BertAttention模块。
        # 对于每个进程，将所有参数初始化出来。
        # Excluded Parameter，不存储在Chunk中的parameter
        # (TODO)模型初始化顺序和optimizer parameter group遍历顺序一致么？
        for name, param in module.named_parameters(recurse=False):
            assert not is_param_registed(param)
            assert param.dtype == torch.half
            print_rank(f'** Converting Params {name}', force=False)

            self.client.append_tensor(param, AccessType.DATA,
                                      ChunkListType.PARAM_FP16, name)

            # Append a tensor to the param fp32 chunk list.
            # Before that, we have to build a fp32 param.
            param_fp32 = torch.nn.Parameter(torch.zeros_like(
                param, dtype=torch.float, device=torch.device('cpu:0')),
                                            requires_grad=False)
            self.client.append_tensor(param_fp32, AccessType.DATA,
                                      ChunkListType.PARAM_FP32, name)
            # Delete the memory of non local tensors
            if not self.client.is_local_tensor(param, AccessType.DATA):
                param.ps_attr._is_local = False
                # TODO(jiaruifang)下面这句将非local的param的内存清零会导致结果错误,
                # 插入这句会影响模型初始化的值。
                if not args.use_fake_dist:
                    param.data = torch.tensor([],
                                              dtype=param.dtype,
                                              device=param.device)
                    param_fp32.data = torch.tensor([],
                                                   dtype=param_fp32.dtype,
                                                   device=param_fp32.device)
            # copy the pytorch data to chunk
            # 者必须在model全部初始化完毕才能调用，因为on-the-fly chunk是无法被access赋值。
            else:
                param.ps_attr._is_local = True
                self.client.access_data(param, torch.device('cpu:0'))
                data_tensor = param.ps_attr.access_tensor(AccessType.DATA)
                data_tensor.copy_(param.data)

                self.client.access_data(param_fp32, torch.device('cpu:0'))
                data_tensor_fp32 = param_fp32.ps_attr.access_tensor(
                    AccessType.DATA)
                data_tensor_fp32.copy_(param.data.float())

                self.client.release_data(param_fp32)
                self.client.release_data(param)
