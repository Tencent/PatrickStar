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

from patrickstar.core import register_param, is_registered, ParamType
from patrickstar.utils import log_dist, get_world_size


# Inserts _post_init_method at the end of init method
# for all sub classes of torch.nn.Module
class InsertPostInitMethodToModuleSubClasses:
    def __enter__(self):
        def preprocess_after(f):
            @functools.wraps(f)
            def wrapper(module, *args, **kwargs):
                f(module, *args, **kwargs)
                self._post_init_method(module)

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
        torch.nn.modules.module.Module._old_init_subclass = (
            torch.nn.modules.module.Module.__init_subclass__
        )

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = classmethod(_init_subclass)

    def __exit__(self, exc_type, exc_value, traceback):
        def _disable_class(cls):
            cls.__init__ = cls._old_init

        # Replace .__init__() for all existing subclasses of torch.nn.Module
        for subclass in torch.nn.modules.module.Module.__subclasses__():
            _disable_class(subclass)

        # Replace .__init__() for future subclasses of torch.nn.Module
        torch.nn.modules.module.Module.__init_subclass__ = (
            torch.nn.modules.module.Module._old_init_subclass
        )

        self._post_context_exec()
        # Now that we cleaned up the metaclass injection, raise the exception.
        if exc_type is not None:
            return False

    # To be implemented by inheriting classes
    def _post_init_method(self, module):
        pass

    def _post_context_exec(self):
        pass


class PSPreProcessCtx(InsertPostInitMethodToModuleSubClasses):
    def __init__(self, client, release_after_init=False):
        self.client = client
        self.param_idx = 0

        self.release_after_init = release_after_init

    def _post_init_method(self, module):
        r"""The function to call at the end of the constructor of each nn.Module.

        The main functionality is registering the params to chunks and
        remove the remote tensor if `release_after_init` is False.
        """
        params = []
        for name, param in module.named_parameters(recurse=False):
            name = f"{module.__class__.__name__}.{name}_{self.param_idx}"
            self.param_idx += 1
            if param.dtype == torch.float:
                register_param(param, ParamType.CHUNK_BASED, name)
                params.append(param)
            else:
                register_param(param, ParamType.TORCH_BASED, name)
                param.ps_attr._is_local = True

        self.client.append_params(params)

        for param in params:
            # Delete the memory of non local tensors
            if self.client.is_local_param(param):
                param.ps_attr._is_local = True
            else:
                param.ps_attr._is_local = False
                # TODO(jiaruifang) fix distributed init bug.
                # Check results will fail when not release_after_init.
                # As release tensor here will make the random seed generator
                # behave differently (less random number generated).
                if not self.release_after_init:
                    # Here we use a non-empty tensor for huggingface. Because it
                    # needs to initialize the weight for padding_idx.
                    param.data = torch.tensor(
                        [0], dtype=torch.float, device=param.device
                    )

    def _post_context_exec(self):
        """The callback function when the context exits.

        1. Copy param.data to fp16 and fp32 chunk based params.
        2. Append dummy chunk so that the number of chunks is an integer multiple of
            number of processes.
        """
        log_dist("Post Model Init Context")

        for chunk in self.client.chunk_list.chunks:
            if chunk.is_local():
                for param in chunk.params:
                    if is_registered(param):
                        init_data = param.data
                        self.client.access(param, torch.device("cpu:0"))
                        param.data.copy_(init_data)
                        fp32_data = self.client.get_fp32(param)
                        fp32_data.copy_(init_data)
                        self.client.release(param)
            else:
                for param in chunk.params:
                    assert not self.client.is_local_param(param)
                    # When release_after_init is True, we will release the remote
                    # param tensor here.
                    # When release_after_init is False, this will help cast dtype of
                    # remote params to torch.half (See the NOTE below).
                    param.data = torch.tensor([], dtype=torch.half, device=param.device)

        num_chunk = len(self.client.chunk_list)
        world_size = get_world_size()
        while num_chunk % world_size != 0:
            self.client.new_dummy_chunk()
            num_chunk += 1
