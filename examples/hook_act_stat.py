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

from patrickstar.utils import get_sys_memory_used, logger

from profiler import profiler


def _update_global_var():
    gpu_mem_used = get_sys_memory_used(
        torch.device(f"cuda:{torch.cuda.current_device()}")
    )
    profiler.timestamp.append(time.time())
    profiler.gpu_memory.append(gpu_mem_used)


# apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(
                module, functional, backward_function, output
            )
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        # logger.debug(f'_apply_to_tensors_only {module}')
        return functional.apply(module, backward_function, outputs)
    else:
        # print('_apply_to_tensors_only', outputs)
        return outputs


class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        logger.debug(f"**After Forward: {ctx.module.__class__.__name__}")
        # TODO(jiaruifang) Why detach?
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        logger.debug(f"**Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        output = output.detach()
        logger.debug(f"**PostBackwardFunction forward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function = pre_backward_function
        return output

    @staticmethod
    def backward(ctx, *args):
        """
        Args:
            activation_grad of the next layer.
        Returns:
            grad of the input activation.
        """
        logger.debug(
            f"**PostBackwardFunction backward: {ctx.module.__class__.__name__}"
        )
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


# Need to be idempotent.
def pre_sub_module_forward_function(sub_module, name):
    # 统计显存
    _update_global_var()


# release submodule
def post_sub_module_forward_function(sub_module, name):
    # 统计显存
    _update_global_var()


def pre_sub_module_backward_function(sub_module, name):
    # 统计显存
    _update_global_var()


def post_sub_module_backward_function(sub_module, name):
    # 统计显存
    _update_global_var()


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
        pre_sub_module_forward_function(module, name)

    def _post_forward_module_hook(module, *args):
        post_sub_module_forward_function(module, name)

    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            pre_sub_module_backward_function(sub_module, name)

        return _apply_to_tensors_only(
            module, PreBackwardFunction, _run_before_backward_function, output
        )

    def _post_backward_module_hook(module, inputs):
        def _run_after_backward_function(sub_module):
            post_sub_module_backward_function(sub_module, name)

        return _apply_to_tensors_only(
            module, PostBackwardFunction, _run_after_backward_function, inputs
        )

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)


def setup_act_stats_hook(module):
    """
    Collect activation statistis during training.
    """
    _register_hooks_recursively(module)
