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

import patrickstar.utils.global_timer as global_timer
from patrickstar.core.parameter import ParamType
from patrickstar.utils import logger, get_rank, get_world_size
from .const import TensorState, AccessType, TrainingStage


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
def pre_sub_module_forward_function(sub_module, client, name):
    flag = False
    rank = get_rank()
    logger.debug(
        f"rank {rank} FWD pre {name}.{sub_module.__class__.__name__} access data"
    )
    for _, param in sub_module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        param.data = client.access_dist(
            param,
            AccessType.DATA,
            client.device,
            client.hook_config["with_mem_saving_comm"],
        )
        flag = True
    if flag:
        client.trigger_memory_tracing()
        client.adjust_chunk_layout()


# release submodule
def post_sub_module_forward_function(sub_module, client, name):
    for sub_name, param in sub_module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        rank = get_rank()
        logger.debug(f"rank {rank} FWD post {name}.{sub_name}")
        if get_world_size() > 1:
            client.release_dist(
                param,
                AccessType.DATA,
                TensorState.HOLD_AFTER_FWD,
                training_stage=TrainingStage.FWD,
                do_allreduce=False,
            )
        else:
            client.release_data(param, TensorState.HOLD_AFTER_FWD)

        if client.training_stage() == TrainingStage.FWD:
            param.ps_attr.fwd_used_cnt += 1

    # client.trigger_memory_tracing()
    # client.adjust_chunk_layout()


def pre_sub_module_backward_function(sub_module, client, name):
    flag = False
    for sub_name, param in sub_module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        rank = get_rank()
        logger.debug(f"rank {rank} BWD pre {name}.{sub_name}")
        if param.ps_attr.data_type == torch.half:
            tmp_tensor = client.access_dist(
                param,
                AccessType.DATA,
                client.device,
                client.hook_config["with_mem_saving_comm"],
            )
            param.data = tmp_tensor

            # NOTE() bwd first visits this param
            if param.ps_attr.bwd_used_cnt == 0:
                param.grad = torch.zeros_like(tmp_tensor)
            param.ps_attr.bwd_used_cnt += 1
        elif param.ps_attr.data_type == torch.float:
            raise RuntimeError("fp32 training is not supported!")
        flag = True
    if flag:
        client.trigger_memory_tracing()
        client.adjust_chunk_layout()


def post_sub_module_backward_function(sub_module, client, name):
    for sub_name, param in sub_module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        # NOTE() We add a fp16 or fp32 attribute to the ps_attr of param.
        # We should not use the data type of param.data in the condition judgment.
        # Since the data type of param.data and ps_attr are not the same.
        # We can change the param.data at will, and there is no restriction to do this.
        assert param.ps_attr.data_type == torch.half
        # NOTE() When a parameter is shared by multiple operators,
        # a reference counter is needed to correctly trigger the chunk reusing.
        # The memory space of the last updated param fp16 is covered by grad fp16.
        client.optimizer.check_overflow(param)
        # NOTE() bwd last visits this pardam
        if param.ps_attr.bwd_used_cnt == param.ps_attr.fwd_used_cnt:
            tmp_tensor = param.ps_attr.access_tensor(AccessType.DATA)
            tmp_tensor.copy_(param.grad)
            if torch.distributed.is_initialized():
                client.release_dist(
                    param,
                    AccessType.DATA,
                    TensorState.HOLD_AFTER_BWD,
                    training_stage=TrainingStage.BWD,
                    do_allreduce=True,
                )
            else:
                client.release_data(param, TensorState.HOLD_AFTER_BWD)
            rank = get_rank()
            logger.debug(f"rank {rank} BWD post before release_dist {name}.{sub_name}")
            param.grad = None

    # client.trigger_memory_tracing()
    # client.adjust_chunk_layout()


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
        pre_sub_module_forward_function(module, client, name)

    def _post_forward_module_hook(module, *args):
        post_sub_module_forward_function(module, client, name)

    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            pre_sub_module_backward_function(sub_module, client, name)

        return _apply_to_tensors_only(
            module, PreBackwardFunction, _run_before_backward_function, output
        )

    def _post_backward_module_hook(module, inputs):
        def _run_after_backward_function(sub_module):
            post_sub_module_backward_function(sub_module, client, name)

        return _apply_to_tensors_only(
            module, PostBackwardFunction, _run_after_backward_function, inputs
        )

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)


def setup_patrickstar_hooks(module, client):
    _register_hooks_recursively(module, client)

    def make_post_backward_hook(param):
        def hook(*ignore):
            client.optimizer.check_overflow(param)
            # Here we use gloo backend group for the cpu tensors (embedding).
            if get_world_size() > 1:
                global_timer.my_timer.start_profile("HOOK_torch_allreduce")
                world_size = get_world_size()
                torch.distributed.all_reduce(
                    param.grad,
                    op=torch.distributed.ReduceOp.SUM,
                    group=client.cpu_comm_group,
                    async_op=False,
                )
                param.grad /= world_size
                global_timer.my_timer.finish_profile("HOOK_torch_allreduce")
            logger.debug(f"rank {get_rank()} allreduce grad {param.ps_attr.name}")

        return hook

    # torch param will not override data with grad,
    # we could use the standard register_hook on them.
    for param in client.torch_param_allreduce_list:
        if param.requires_grad:
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(make_post_backward_hook(param))
            client.grad_accs.append(grad_acc)
