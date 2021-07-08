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
from patrickstar.core import PatrickStarClient, AccessType, ChunkListType
from patrickstar.core import PSParameter, register_param, is_param_registed, register_torch_param
from patrickstar.deepspeed_helper.global_vars import get_args

_orig_torch_empty = torch.empty


# 直接在初始化后把，内存放在ps tensor上
def empty_cuda_tensor_half(*size, **kwargs):
    if not 'device' in kwargs.keys():
        kwargs['device'] = torch.device('cpu:0')
    tensor = _orig_torch_empty(*size, **kwargs)
    if tensor.is_floating_point():
        return tensor.half()
    else:
        return tensor


# TODO能否和param的注册过程放在一起
# 问题是看不到param
def new_cuda_tensor_half(cls, *args):
    device = torch.device('cpu:0')
    tensor = torch.ones((1, 1), device=device).new_empty(*args).half()
    print_rank(
        f'During model initialization, a new tensor of shape {tensor.shape} is created.'
    )
    if tensor.is_floating_point():
        return tensor.half()
    else:
        return tensor


def empty_cuda_tensor(*size, **kwargs):
    if not 'device' in kwargs.keys():
        kwargs['device'] = torch.device('cpu:0')
    tensor = _orig_torch_empty(*size, **kwargs)
    return tensor


def new_cuda_tensor(cls, *args):
    device = torch.device('cpu:0')
    tensor = torch.ones((1, 1), device=device).new_empty(*args)
    return tensor


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self,
                 enabled=True,
                 mem_efficient_linear=True,
                 config=None,
                 dtype=None):
        self.mem_efficient_linear = mem_efficient_linear
        self.enabled = enabled
        self._set_dtype(config, dtype)
        assert self.dtype in [
            torch.half, torch.float
        ], f"Invalid data type {self.dtype}, allowed values are [torch.half, torch.float]"

    def __enter__(self):
        if not self.enabled:
            return

        def partition_after(f):
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
            cls.__init__ = partition_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = partition_after(cls.__init__)

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
            torch.Tensor.__new__ = new_cuda_tensor_half
            torch.empty = empty_cuda_tensor_half
        else:
            torch.Tensor.__new__ = new_cuda_tensor
            torch.empty = empty_cuda_tensor

    def __exit__(self, exc_type, exc_value, traceback):
        if not self.enabled:
            return

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


# Replaces all parameters in module with Scattered Parameters
class Init(InsertPostInitMethodToModuleSubClasses):
    def __init__(self,
                 module=None,
                 client: PatrickStarClient = None,
                 data_parallel_group=None,
                 mem_efficient_linear=True,
                 remote_device=None,
                 pin_memory=False,
                 config=None,
                 enabled=True,
                 dtype=None):
        super().__init__(enabled=enabled,
                         mem_efficient_linear=mem_efficient_linear,
                         config=config,
                         dtype=dtype)
        # TODO backend is not locked to nccl
        if not torch.distributed.is_initialized():
            assert torch.distributed.is_initialized(
            ), "Parameters cannot be scattered without initializing torch.distributed"
        if data_parallel_group is None:
            self.ps_process_group = torch.distributed.group.WORLD
        else:
            self.ps_process_group = data_parallel_group

        self.rank = torch.distributed.get_rank(group=self.ps_process_group)
        self.world_size = torch.distributed.get_world_size(
            group=self.ps_process_group)

        self._client = client

    def _post_init_method(self, module):
        """
        在构造model过程中的每个sub_module构造完毕后执行
        1. 保留本proc管理的模型内存，删除其他进程的模型。
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
            assert param.dtype == torch.float
            print_rank(f'** Converting Params {name}', force=False)

            register_param(param, name)
            numel = param.ps_attr.ps_numel
            data_type = torch.half

            chunk_index_in_group, chunk_id = self._client.chunk_schema_scheduler.add_tensor(
                param.ps_attr.data_id(), numel, param, AccessType.DATA,
                data_type, ChunkListType.PARAM_FP16)
            # 将不属于本地Chunk param的data tensor删除掉
            if chunk_index_in_group != rank:
                param.ps_attr._is_local = False
                # TODO(jiaruifang)下面这句将非local的param的内存清零会导致结果错误,
                # 插入这句会影响模型初始化的值。
                if not args.use_fake_dist:
                    param.data = torch.zeros(1,
                                             dtype=param.dtype,
                                             device=param.device)
            else:
                param.ps_attr._is_local = True
