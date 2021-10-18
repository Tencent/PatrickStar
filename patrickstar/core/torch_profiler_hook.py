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
import time
import torch

from patrickstar.core.hook import (
    PreBackwardFunction,
    PostBackwardFunction,
    _apply_to_tensors_only,
)
from patrickstar.utils import get_sys_memory_used, logger

from patrickstar.profiler import torch_profiler as profiler


def _update_global_var():
    gpu_mem_used = get_sys_memory_used(
        torch.device(f"cuda:{torch.cuda.current_device()}")
    )
    profiler.timestamp.append(time.time())
    profiler.gpu_memory.append(gpu_mem_used)


def _register_hooks_recursively(module, name=""):
    r"""Register hook in post order traverse."""

    for child_name, child in module.named_children():
        logger.debug(f"{child.__class__.__name__}")
        _register_hooks_recursively(child, name + child_name)

    # Early return on modules with no parameters or buffers that
    # are not in their children.
    if (
        len(list(module.named_parameters(recurse=False))) == 0
        and len(list(module.named_buffers(recurse=False))) == 0
    ):
        return

    def _pre_forward_module_hook(module, *args):
        _update_global_var()

    def _post_forward_module_hook(module, *args):
        _update_global_var()

    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            _update_global_var()

        return _apply_to_tensors_only(
            module, PreBackwardFunction, _run_before_backward_function, output
        )

    def _post_backward_module_hook(module, inputs):
        def _run_after_backward_function(sub_module):
            _update_global_var()

        return _apply_to_tensors_only(
            module, PostBackwardFunction, _run_after_backward_function, inputs
        )

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)


def register_torch_profiler_hook(module):
    """
    Collect activation statistis during training.
    """
    _register_hooks_recursively(module)
