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
from utils import init_distributed, see_memory_usage
import torch
import functools
from utils import logger, print_rank, debug_flag
from client.parameter import PSParameter, register_param

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

        #un doing it here will undo it during training
        #if self.mem_efficient_linear:
        #    torch.nn.functional.linear = self.linear_bk
        #        if self.mem_efficient_linear:
        #            torch.nn.functional.linear = self.linear_bk

        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _set_dtype(self, ds_config, dtype):
        if ds_config is not None and dtype is None:
            _ds_config = DeepSpeedConfig(ds_config)
            self.dtype = torch.half if _ds_config.fp16_enabled else torch.float
        elif dtype is None:
            self.dtype = torch.half
        else:
            self.dtype = dtype


# Replaces all parameters in module with Scattered Parameters
class Init(InsertPostInitMethodToModuleSubClasses):
    param_id = 0

    def __init__(self,
                 module=None,
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
            init_distributed(dist_backend="gloo" if debug_flag else "nccl")
            assert torch.distributed.is_initialized(
            ), "Parameters cannot be scattered without initializing torch.distributed"
        if data_parallel_group is None:
            self.ps_process_group = torch.distributed.group.WORLD
        else:
            self.ps_process_group = data_parallel_group

        self.rank = torch.distributed.get_rank(group=self.ps_process_group)
        self.world_size = torch.distributed.get_world_size(
            group=self.ps_process_group)

        #Local device is the device where the parameters are consumed
        #It is the device where parameters are fully instantiated using allgather
        self.local_device = torch.device('cuda:{}'.format(
            os.environ["LOCAL_RANK"]))

        #Remote device is the device where parameter partiitons are stored
        #It can be same as local_device or it could be CPU or NVMe.
        self.remote_device = torch.device('cpu:0')

        # If we are provided an already-allocated module to prepare.
        # if module is not None:
        #     assert isinstance(module, torch.nn.Module)
        #     for param in module.parameters(recurse=True):
        #         if is_zero_param(param):
        #             continue
        #         self._convert_to_deepspeed_param(param)
        #         param.partition()

    def _post_init_method(self, module):
        """
        在构造model过程中的每个sub_module构造完毕后执行
        1. 保留rank=0的模型内存，删除其他rank的模型内存。
        2.
        """
        #see_memory_usage(f"Before converting parmas in {module.__class__.__name__}", force=False)
        print_rank(f'Converting Params in {module.__class__.__name__}',
                   force=True)
        for name, param in module.named_parameters(recurse=False):
            register_param(param, name)
            #TODO(jiaruifang) 只有rank 0载入模型
            # if self.rank != 0:
            #     param.data = torch.zeros(1, device = torch.device('cpu:0'))

        # see_memory_usage(
        #     f"Before converting and partitioning parmas in {module.__class__.__name__}",
        #     force=True)

        # see_memory_usage(
        #     f"After converting and partitioning parmas in {module.__class__.__name__}",
        #     force=True)

    def _aligned_size(self, param):
        return param.ds_numel + self._padding_size(param)

    def _padding_size(self, param):
        remainder = param.ds_numel % self.world_size
        return (self.world_size - remainder) if remainder else 0
