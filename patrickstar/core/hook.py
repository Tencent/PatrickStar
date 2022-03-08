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

from patrickstar.core.const import TensorState
import torch

from patrickstar.utils import logger, get_rank, get_world_size, global_timer


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
        # TODO(jiaruifang) Why detach?
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        # NOTE(zilinzhu) the ref count for modules that will run
        # multiple times in one iteration.
        if output.requires_grad:
            module.ds_grads_remaining += 1
            ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        return output

    @staticmethod
    def backward(ctx, *args):
        ctx.module.ds_grads_remaining -= 1
        if ctx.module.ds_grads_remaining == 0:
            ctx.pre_backward_function(ctx.module)
        return (None, None) + args


def load_params(module, client, name):
    flag = False
    for _, param in module.named_parameters(recurse=False):
        if param.ps_attr.is_chunk_based():
            client.access_dist(param, client.device)
            flag = True
    if flag:
        client.mem_tracer.trace()


# release submodule
def offload_params(module, client, name):
    for param in module.parameters(recurse=False):
        if param.ps_attr.is_chunk_based():
            client.release(param)


def reduce_grad(param, client):
    # TODO(zilinzhu) Here we may do allreduce on torch based params twice.
    if param.grad is None:
        return
    if param.ps_attr.is_chunk_based() and param.ps_attr.state != TensorState.COMPUTE:
        return
    client.optimizer.check_overflow(param)
    chunk_id = param.ps_attr.info.chunk_id
    chunk = client.chunk_list[chunk_id]
    world_size = get_world_size()
    if world_size > 1:
        global_timer.start_profile("HOOK_torch_allreduce")
        if param.ps_attr.is_chunk_based():
            dst = chunk.comm_info.offset
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
        else:
            torch.distributed.all_reduce(
                param.grad,
                op=torch.distributed.ReduceOp.SUM,
                async_op=False,
            )
            param.grad /= world_size
        global_timer.finish_profile("HOOK_torch_allreduce")
    if param.ps_attr.is_chunk_based():
        grad = client.get_grad(param, torch.device("cpu:0"))
        grad.copy_(param.grad)
        param.grad = None


def post_module_backward_function(module, client, name):
    for param in module.parameters(recurse=False):
        reduce_grad(param, client)
    offload_params(module, client, name)


def _register_hooks_recursively(module, client, name=""):
    r"""Register hook in post order traverse."""

    for child_name, child in module.named_children():
        logger.debug(f"{child.__class__.__name__}")
        _register_hooks_recursively(child, client, name + child_name)

    # Early return on modules with no parameters or buffers that
    # are not in their children.
    if (
        len(list(module.parameters(recurse=False))) == 0
        and len(list(module.buffers(recurse=False))) == 0
    ):
        return

    def _pre_forward_module_hook(module, *args):
        load_params(module, client, name)

    def _post_forward_module_hook(module, *args):
        offload_params(module, client, name)

    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            load_params(sub_module, client, name)

        return _apply_to_tensors_only(
            module, PreBackwardFunction, _run_before_backward_function, output
        )

    def _post_backward_module_hook(module, inputs):
        module.ds_grads_remaining = 0

        def _run_after_backward_function(sub_module):
            if sub_module.ds_grads_remaining == 0:
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
