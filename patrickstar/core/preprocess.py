# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import contextlib
import functools

import torch

from patrickstar.core import PatrickStarClient, AccessType, ChunkType
from patrickstar.core import register_param, is_param_registered, ParamType
from patrickstar.manager import _runtime_config
from patrickstar.ops import Embedding
from patrickstar.utils import logger, log_dist, print_rank, get_rank, get_world_size
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


@contextlib.contextmanager
def torch_scope(do_allreduce=True):
    r"""All parameters initialized in this scope will not be managed in chunks."""
    _runtime_config.push()
    _runtime_config.config["use_chunk"] = False
    _runtime_config.config["do_allreduce"] = do_allreduce
    yield
    _runtime_config.pop()


def cast_forward(module, dtype):
    if not isinstance(dtype, torch.dtype):
        raise ValueError("dtype should be of torch.dtype.")

    old_forward = module.forward

    def forward(*args, **kwargs):
        casted_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor) and torch.is_floating_point(arg):
                casted_args.append(arg.to(dtype))
            else:
                casted_args.append(arg)
        casted_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
                casted_kwargs[k] = v.to(dtype)
            else:
                casted_kwargs[k] = v
        return old_forward(*casted_args, **casted_kwargs)

    module.forward = forward


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
        not_init=False,
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
        self.not_init = not_init

    def _pre_context_exec(self):
        Embedding.use_cpu = self.use_cpu_embedding

        def _new(cls, *args, **kwargs):
            embedding = object.__new__(Embedding)
            return embedding

        torch.nn.Embedding.__new__ = _new

    def _post_context_exec(self):
        """The callback function when the context exits.

        1. Copy param.data to fp16 and fp32 chunk based params.
        2. Append dummy chunk so that the number of chunks is an integer multiple of
            number of processes.
        3. Add a dummy param at the start of CPU Embedding for huggingface.
        """
        log_dist("Post Model Init Context")

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
            self.client.chunk_ids_generator(ChunkType.PARAM_FP16),
            self.client.chunk_ids_generator(ChunkType.PARAM_FP32),
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
                    if not self.not_init:
                        if is_param_registered(param_fp32) and is_param_registered(
                            param_fp16
                        ):
                            ps_data_fp16 = self.client.access_data(
                                param_fp16, torch.device("cpu:0")
                            )

                            ps_data_fp32 = self.client.access_data(
                                param_fp32, torch.device("cpu:0")
                            )

                            # Here the dtype of param_fp16 is actually fp32.
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
        log_dist(f"Param fp16 chunk num {chunk_num}")
        while chunk_num % world_size != 0:
            self.client.append_dummy_chunk(torch.half, ChunkType.PARAM_FP16)
            chunk_num += 1

    def _post_init_method(self, module):
        r"""The function to call at the end of the constructor of each nn.Module.

        The main functionality is registering the params to chunks and
        remove the remote tensor if `release_after_init` is False.
        """
        self.submodule_id += 1
        see_memory_usage(
            f"Before converting parmas in {module.__class__.__name__}", force=False
        )
        if self.use_cpu_embedding:
            # In CPU embedding optimization,
            # the embedding is managed by torch instead of chunk.
            if module.__class__.__name__ == "Embedding":
                logger.debug(
                    f"** Converting Maintain PyTorch Params in {module.__class__.__name__}"
                )
                for name, param in module.named_parameters(recurse=False):
                    param_fp32 = torch.nn.Parameter(param.data.clone())
                    register_param(
                        param, ParamType.TORCH_BASED, torch.float, f"embedding_{name}"
                    )
                    self.client.torch_param_allreduce_list.append(param)
                return

        if not _runtime_config.use_chunk:
            for name, param in module.named_parameters(recurse=False):
                name = f"{module.__class__.__name__}.{name}_{self.param_idx}"
                register_param(param, ParamType.TORCH_BASED, torch.float, name)
                if _runtime_config.do_allreduce:
                    self.client.torch_param_allreduce_list.append(param)

            # We need to cast the inputs to fp32 for the unmanaged modules.
            cast_forward(module, torch.float)
            return

        # The granularity of `post_init_method` is nn.Module, e.g. BertAttention.
        # For every process, we will initialize the params.
        # NOTE() The order of params in model initialization is a bit different from the
        # the one in optimizer parameter group.
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
            param_fp16_list, torch.half, AccessType.DATA, ChunkType.PARAM_FP16
        )
        self.client.append_tensor(
            param_fp32_list, torch.float, AccessType.DATA, ChunkType.PARAM_FP32
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

        cast_forward(module, torch.half)
