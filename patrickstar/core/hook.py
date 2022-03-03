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

import torch

from patrickstar.core.const import TensorState, TrainingStage
from patrickstar.core.parameter import ParamType
from patrickstar.utils import logger, get_rank, get_world_size, global_timer


# For each tensor in outputs run the forward_funciton and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(
    module, forward_function, backward_function, outputs
):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(
                module, forward_function, backward_function, output
            )
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            logger.debug("output require grad {outputs.shape}")
            outputs.register_hook(backward_function)
        else:
            logger.debug("output dose not require grad {outputs}")
        return outputs
    else:
        return outputs


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
        return functional.apply(module, backward_function, outputs)
    else:
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
def pre_module_forward_function(module, client, name):
    flag = False
    rank = get_rank()
    logger.debug(f"rank {rank} FWD pre {name}.{module.__class__.__name__} access data")
    for _, param in module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        param.data = client.access_dist(
            param,
            client.device,
        )
        flag = True
    if flag:
        client.trigger_memory_tracing()


# release submodule
def post_module_forward_function(module, client, name):
    for sub_name, param in module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        rank = get_rank()
        logger.debug(f"rank {rank} FWD post {name}.{sub_name}")
        if get_world_size() > 1:
            client.release_dist(
                param,
                TensorState.HOLD_AFTER_FWD,
                training_stage=TrainingStage.FWD,
            )
        else:
            client.release(param, TensorState.HOLD_AFTER_FWD)


def pre_module_backward_function(module, client, name):
    flag = False
    for sub_name, param in module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        rank = get_rank()
        logger.debug(f"rank {rank} BWD pre {name}.{sub_name}")
        tmp_tensor = client.access_dist(
            param,
            client.device,
        )
        param.data = tmp_tensor
        flag = True
    if flag:
        client.trigger_memory_tracing()


def reduce_grad(param, client):
    chunk_id = param.ps_attr.info.chunk_id
    chunk = client.chunk_list[chunk_id]
    dst = chunk.comm_info.offset
    # Here we use gloo backend group for the cpu tensors (embedding).

    world_size = get_world_size()
    if world_size > 1:
        global_timer.my_timer.start_profile("HOOK_torch_allreduce")
        torch.distributed.reduce(
            param.grad,
            dst,
            op=torch.distributed.ReduceOp.SUM,
            async_op=False,
        )
        if dst == get_rank():
            param.grad /= world_size
        else:
            param.grad = None
        global_timer.my_timer.finish_profile("HOOK_torch_allreduce")
    logger.debug(f"rank {get_rank()} allreduce grad {param.ps_attr.name}")


def post_module_backward_function(module, client, name):
    for _, param in module.named_parameters(recurse=False):
        reduce_grad(param, client)
    for sub_name, param in module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        assert param.ps_attr.dtype == torch.float
        # NOTE() When a parameter is shared by multiple operators,
        # a reference counter is needed to correctly trigger the chunk reusing.
        # The memory space of the last updated param fp16 is covered by grad fp16.
        if torch.distributed.is_initialized():
            client.release_dist(
                param,
                TensorState.HOLD_AFTER_BWD,
                training_stage=TrainingStage.BWD,
            )
        else:
            client.release(param, TensorState.HOLD_AFTER_BWD)
        rank = get_rank()
        logger.debug(f"rank {rank} BWD post before release_dist {name}.{sub_name}")


def _register_hooks_recursively(module, client, name=""):
    r"""Register hook in post order traverse."""

    for child_name, child in module.named_children():
        logger.debug(f"{child.__class__.__name__}")
        _register_hooks_recursively(child, client, name + child_name)

    # Early return on modules with no parameters or buffers that
    # are not in their children.
    if (
        len(list(module.named_parameters(recurse=False))) == 0
        and len(list(module.named_buffers(recurse=False))) == 0
    ):
        return

    def _pre_forward_module_hook(module, *args):
        pre_module_forward_function(module, client, name)

    def _post_forward_module_hook(module, *args):
        post_module_forward_function(module, client, name)

    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            pre_module_backward_function(sub_module, client, name)

        return _apply_to_tensors_only(
            module, PreBackwardFunction, _run_before_backward_function, output
        )

    def _post_backward_module_hook(module, inputs):
        def _run_after_backward_function(sub_module):
            post_module_backward_function(sub_module, client, name)

        return _apply_to_tensors_only(
            module, PostBackwardFunction, _run_after_backward_function, inputs
        )

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)


def setup_patrickstar_hooks(module, client):
    _register_hooks_recursively(module, client)
