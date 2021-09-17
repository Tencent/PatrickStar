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

import functools

import torch

from patrickstar.core import PatrickStarClient, AccessType, ChunkListType
from patrickstar.core import register_param, is_param_registered, ParamType
from patrickstar.ops import Embedding
from patrickstar.utils import logger, print_rank, get_rank, get_world_size
from patrickstar.utils import see_memory_usage

_orig_torch_empty = torch.empty


def empty_cpu_tensor_half(*size, **kwargs):
    if "device" not in kwargs.keys():
        kwargs["device"] = torch.device("cpu:0")
    tensor = _orig_torch_empty(*size, **kwargs)
    if tensor.is_floating_point():
        return tensor.half()
    else:
        return tensor


def new_cpu_tensor_half(cls, *args):
    device = torch.device("cpu:0")
    tensor = torch.ones((1, 1), device=device).new_empty(*args).half()
    print_rank(
        f"During model initialization, a new tensor of shape {tensor.shape} is created."
    )
    if tensor.is_floating_point():
        return tensor.half()
    else:
        return tensor


def empty_cpu_tensor(*size, **kwargs):
    if "device" not in kwargs.keys():
        kwargs["device"] = torch.device("cpu:0")
    tensor = _orig_torch_empty(*size, **kwargs)
    return tensor


def new_cpu_tensor(cls, *args):
    device = torch.device("cpu:0")
    tensor = torch.ones((1, 1), device=device).new_empty(*args)
    return tensor


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses(object):
    def __init__(self, config=None, dtype=None):
        self._set_dtype(config, dtype)
        assert self.dtype in [
            torch.half,
            torch.float,
        ], f"Invalid data type {self.dtype}, allowed values are [torch.half, torch.float]"

    def __enter__(self):
        def preprocess_after(f):
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):
                print_rank(
                    f"Before initializing {module.__class__.__name__}", force=False
                )
                f(module, *args, **kwargs)
                self._post_init_method(module)
                print_rank(
                    f"After initializing followed by post init for {module.__class__.__name__}",
                    force=False,
                )

            return wrapper

        def _enable_class(cls):
            cls._old_init = cls.__init__
            cls.__init__ = preprocess_after(cls.__init__)

        def _init_subclass(cls, **kwargs):
            cls.__init__ = preprocess_after(cls.__init__)

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            if subclass == torch.nn.Embedding:
                _enable_class(Embedding)
            else:
                _enable_class(subclass)

        # holding on to the current __init__subclass__ for exit
        torch.nn.modules.module.Module._old_init_subclass = (
            torch.nn.modules.module.Module.__init_subclass__
        )
        torch.Tensor.__old_new__ = torch.Tensor.__new__

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)
        if self.dtype == torch.half:
            torch.Tensor.__new__ = new_cpu_tensor_half
            torch.empty = empty_cpu_tensor_half
        else:
            torch.Tensor.__new__ = new_cpu_tensor
            torch.empty = empty_cpu_tensor

        self._pre_context_exec()

    def __exit__(self, exc_type, exc_value, traceback):
        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            if subclass == torch.nn.Embedding:
                _disable_class(Embedding)
            else:
                _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = (
            torch.nn.modules.module.Module._old_init_subclass
        )

        torch.Tensor.__new__ = torch.Tensor.__old_new__
        torch.empty = _orig_torch_empty

        self._post_context_exec()
        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _pre_context_exec(self):
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

    def __init__(
        self,
        client: PatrickStarClient,
        release_after_init=False,
        use_cpu_embedding=False,
        dtype=None,
    ):
        super().__init__(config=None, dtype=dtype)
        self.rank = get_rank()
        self.world_size = get_world_size()
        self.client = client
        self.dummy_param_list = []
        self.param_idx = 0

        self.release_after_init = release_after_init
        self.use_cpu_embedding = use_cpu_embedding

        self.submodule_id = -1

    def _pre_context_exec(self):
        Embedding.use_cpu = self.use_cpu_embedding

        def _new(cls, *args, **kwargs):
            embedding = object.__new__(Embedding)
            return embedding

        torch.nn.Embedding.__new__ = _new

    def _post_context_exec(self):
        """
        初始化context退出时执行本函数
        1. 拷贝param的data，到param fp32和param fp16中去
        2. append dummy chunk，使chunk num是进程数的整数倍 TODO(jiaruifang)
        """
        logger.info("Post Model Init Context")

        def _origin_new(cls, *arg, **kwargs):
            return object.__new__(cls)

        torch.nn.Embedding.__new__ = _origin_new

        if Embedding.use_cpu:
            for instance in Embedding.instances:
                # A walkaround for huggingface.
                # Huggingface will use the type of the first parameter as the
                # dtype of the module. And we need the module to be identified as
                # fp16 for the mixed precision training in patrickstar.
                # However, when use_cpu_embedding is True, the weight of embedding
                # remains to fp32 (otherwise cause error on older version of pytorch).
                # As the embedding is usually the first submodule, we insert a
                # dummy fp16 Parameter as the placeholder.
                #
                # TODO(zilinzhu) Figure out why dummy in the __init__ of Embedding will
                # cause numeric error.
                instance.dummy = torch.nn.Parameter(
                    torch.tensor([], dtype=torch.half), requires_grad=False
                )
                register_param(
                    instance.dummy,
                    ParamType.TORCH_BASED,
                    torch.half,
                    "embedding_dummy",
                )
                instance._parameters.move_to_end("dummy", last=False)
            # Clean the members to prevent elements not grabage collected.
            Embedding.instances = []
            Embedding.use_cpu = False

        chunk_num = 0
        for param_fp16_chunk_id, param_fp32_chunk_id in zip(
            self.client.chunk_ids_generator(ChunkListType.PARAM_FP16),
            self.client.chunk_ids_generator(ChunkListType.PARAM_FP32),
        ):
            if self.client.chunk_tensor_index.is_local_chunk(param_fp16_chunk_id):
                for param_fp16, param_fp32 in zip(
                    self.client.chunk_tensor_index.params_generator(
                        param_fp16_chunk_id
                    ),
                    self.client.chunk_tensor_index.params_generator(
                        param_fp32_chunk_id
                    ),
                ):
                    if is_param_registered(param_fp32) and is_param_registered(
                        param_fp16
                    ):
                        ps_data_fp16 = self.client.access_data(
                            param_fp16, torch.device("cpu:0")
                        )

                        ps_data_fp32 = self.client.access_data(
                            param_fp32, torch.device("cpu:0")
                        )

                        # param_fp16目前还是fp32
                        ps_data_fp16.copy_(param_fp16.data)
                        ps_data_fp32.copy_(param_fp16.data)

                        self.client.release_data(param_fp16)
                        self.client.release_data(param_fp32)
                        param_fp16 = param_fp16.to(torch.half)
            else:
                for param_fp16 in self.client.chunk_tensor_index.params_generator(
                    param_fp16_chunk_id
                ):
                    assert not self.client.is_local_param(param_fp16, AccessType.DATA)
                    # When release_after_init is True, we will release the remote
                    # param tensor here.
                    # When release_after_init is False, this will help cast dtype of
                    # remote params to torch.half (See the NOTE below).
                    param_fp16.data = torch.tensor(
                        [], dtype=torch.half, device=param_fp16.device
                    )
            chunk_num += 1

        world_size = get_world_size()
        logger.info(f"param fp16 chunk num {chunk_num}")
        while chunk_num % world_size != 0:
            self.client.append_dummy_chunk(torch.half, ChunkListType.PARAM_FP16)
            chunk_num += 1

    def _post_init_method(self, module):
        """
        在model的param被PyTorch初始化完毕后完成
        1. 保留local的tensor，通过删除remote tensor的方式
        2. 将model param拷贝到chunk对应的内存中
        """
        self.submodule_id += 1
        see_memory_usage(
            f"Before converting parmas in {module.__class__.__name__}", force=False
        )
        if self.use_cpu_embedding:
            # cpu_embedding优化把embedding交给Torch管理而非Chunk
            if module.__class__.__name__ == "Embedding":
                logger.debug(
                    f"** Converting Maintain PyTorch Params in {module.__class__.__name__}"
                )
                for name, param in module.named_parameters(recurse=False):
                    param_fp32 = torch.nn.Parameter(param.data.clone())
                    register_param(
                        param, ParamType.TORCH_BASED, torch.float, f"embedding_{name}"
                    )
                    self.client.torch_param_list.append(param)
                return

        print_rank(f"Converting Params in {module.__class__.__name__}", force=False)

        # 在模型初始化的过程构造模型，post_init_method调用粒度是一个SubModule，比如BertAttention模块。
        # 对于每个进程，将所有参数初始化出来。
        # (NOTE)模型初始化顺序和optimizer parameter group遍历顺序虽不一致，但很相似
        param_fp16_list = []
        param_fp32_list = []
        for name, param in module.named_parameters(recurse=False):
            name = f"{module.__class__.__name__}.{name}_{self.param_idx}"
            register_param(param, ParamType.CHUNK_BASED, torch.half, name)
            self.param_idx += 1
            param_fp16_list.append(param)
            logger.debug(
                f"** Converting Params {name} in module id {self.submodule_id}"
            )
            # Append a tensor to the param fp32 chunk list.
            # Before that, we have to build a fp32 param.
            param_fp32 = torch.nn.Parameter(
                torch.tensor([], dtype=torch.float, device=torch.device("cpu:0")),
                requires_grad=False,
            )
            param_fp32_list.append(param_fp32)
            register_param(
                param_fp32, ParamType.CHUNK_BASED, torch.float, f"{name}_fp32"
            )
            param_fp32.ps_attr.reset_shape(param.shape)
            self.client.param_fp16_to_param_fp32_map[param] = param_fp32
            self.client.chunk_based_param_fp16.append(param)

        self.client.append_tensor(
            param_fp16_list, torch.half, AccessType.DATA, ChunkListType.PARAM_FP16
        )
        self.client.append_tensor(
            param_fp32_list, torch.float, AccessType.DATA, ChunkListType.PARAM_FP32
        )

        for param_fp16, param_fp32 in zip(param_fp16_list, param_fp32_list):
            # Delete the memory of non local tensors
            if not self.client.is_local_param(param_fp16, AccessType.DATA):
                param_fp16.ps_attr._is_local = False
                param_fp32.ps_attr._is_local = False
                # TODO(jiaruifang) fix distributed init bug.
                # Check results will fail when not release_after_init.
                # As release tensor here will make the random seed generator
                # behave differently (less random number generated).
                # NOTE(why dtype is torch.float rather than torch.half)
                # PyTorch version lower than 1.5 can not initialize torch.half
                # tensors on CPU. So although the param_fp16 is a fp16 type param,
                # its pytorch dtype still be float.
                if not self.release_after_init:
                    # Here we use a non-empty tensor for huggingface. Because it
                    # needs to initialize the weight for padding_idx.
                    param_fp16.data = torch.tensor(
                        [0], dtype=torch.float, device=param_fp16.device
                    )
            else:
                param_fp16.ps_attr._is_local = True
                param_fp32.ps_attr._is_local = True
