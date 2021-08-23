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

        self._post_context_exec()
        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _post_context_exec(self):
        pass

    def _set_dtype(self, ds_config, dtype):
        if dtype is None:
            self.dtype = torch.half
        else:
            self.dtype = dtype


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
        self.param_idx = 0

    def _post_context_exec(self):
        """
        初始化context退出时执行本函数
        1. 拷贝param的data，到param fp32和param fp16中去
        1. 补充dummy chunk，使chunk num是进程数的整数倍
        """
        print('_post_context_exec')
        for param_fp16_chunk_id, param_fp32_chunk_id in zip(
                self.client.generate_chunk_ids(ChunkListType.PARAM_FP16),
                self.client.generate_chunk_ids(ChunkListType.PARAM_FP32)):
            for param_fp16, param_fp32 in zip(
                    self.client.chunk_tensor_index.generate_params(
                        param_fp16_chunk_id),
                    self.client.chunk_tensor_index.generate_params(
                        param_fp32_chunk_id)):
                self.client.access_data(param_fp16, torch.device('cpu:0'))
                ps_data_fp16 = param_fp16.ps_attr.access_tensor(
                    AccessType.DATA)

                self.client.access_data(param_fp32, torch.device('cpu:0'))
                ps_data_fp32 = param_fp32.ps_attr.access_tensor(
                    AccessType.DATA)

                ps_data_fp16.copy_(param_fp16.data)
                ps_data_fp32.copy_(param_fp16.data)

                self.client.release_data(param_fp16)
                self.client.release_data(param_fp32)

    def _is_local_param(self, param, access_type):
        """
        TODO(jiaruifang)暂时和client中的is_local_param重复，未来会合并
        """
        args = get_args()
        chunk_id = self.client.chunk_tensor_index.get_chunk_id(
            param, access_type)
        comm_group_id = self.client.chunk_tensor_index.dict_chunk_id_comm_group_id[
            chunk_id]
        return args.local_rank == comm_group_id

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
        # (NOTE)模型初始化顺序和optimizer parameter group遍历顺序虽不一致，但很相似
        for name, param in module.named_parameters(recurse=False):
            assert not is_param_registed(param)
            name = f'{name}_{self.param_idx}'
            self.param_idx += 1
            logger.info(f'** Converting Params {name}')
            self.client.append_tensor(param, torch.half, AccessType.DATA,
                                      ChunkListType.PARAM_FP16, f'{name}_fp16')

            # Append a tensor to the param fp32 chunk list.
            # Before that, we have to build a fp32 param.
            param_fp32 = torch.nn.Parameter(torch.tensor(
                [], dtype=torch.float, device=torch.device('cpu:0')),
                                            requires_grad=False)
            register_param(param_fp32, f'{name}_fp32')
            param_fp32.ps_attr.reset_shape(param.shape)
            self.client.append_tensor(param_fp32, torch.float, AccessType.DATA,
                                      ChunkListType.PARAM_FP32, f'{name}_fp32')

            # Delete the memory of non local tensors
            if not self._is_local_param(param, AccessType.DATA):
                param.ps_attr._is_local = False
            else:
                param.ps_attr._is_local = True
