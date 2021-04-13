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

import torch
import logging

############# HOOKS ####################


# 可以修改outputs
class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        logging.log(logging.DEBUG,
                    f"After Forward: {ctx.module.__class__.__name__}")
        # why detach?detach后给下一层作为输入，似乎没有用，fwd输出都会用backward作为反向
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        logging.log(logging.DEBUG,
                    f"Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        if output.requires_grad:
            #TODO SOME TIMES post backward does not seem to be triggered debug in detail
            #Should only cause increase in memory not correctness issue
            #if output.grad_fn.__class__.__name__ == 'ViewBackward':
            #    ctx.view=True
            #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
            #assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
            #if module.ds_grads_remaining == 0:
            #    print(f"Before Forward: {ctx.module.__class__.__name__}")
            module.ds_grads_remaining += 1
            ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        return output

    # arguments: 下一层的activation_grad, rets: 输入activation的grad
    @staticmethod
    def backward(ctx, *args):
        ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        if ctx.module.ds_grads_remaining == 0:
            ctx.pre_backward_function(ctx.module)
            logging.log(logging.DEBUG,
                        f"After Backward: {ctx.module.__class__.__name__}")
        # why (None, None) as first two returns
        else:
            logging.log(
                logging.DEBUG,
                f"After Backward: {ctx.module.__class__.__name__} None, None")
        return (None, None) + args


#apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module, functional,
                                                    backward_function, output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        return functional.apply(module, backward_function, outputs)
    else:
        return outputs


# 必须具备重复调用，第二次无效的能力 fetch submodule
def pre_sub_module_forward_function(sub_module, client):
    logging.log(
        logging.DEBUG,
        f'{sub_module.__class__.__name__} pre_sub_module_forward_function, access HybridPS get param'
    )
    for param in sub_module.parameters(recurse=False):
        client.access_data(param, torch.device('cuda:0'))


# release submodule
def post_sub_module_forward_function(sub_module, client):
    logging.log(
        logging.DEBUG,
        f'{sub_module.__class__.__name__} post_sub_module_forward_function, access HybridPS get param'
    )
    for param in sub_module.parameters(recurse=False):
        client.release_data(param)


def pre_sub_module_backward_function(sub_module, client):
    # TODO(jiaruifang) backward前处理逻辑
    logging.log(
        logging.DEBUG,
        f'Before sub module backward function {sub_module.__class__.__name__} allgather'
    )
    # TODO
    for param in sub_module.parameters(recurse=False):
        client.access_data(param, torch.device('cuda:0'))
        client.access_grad(param, torch.device('cuda:0'))


# release param of submodule
def post_sub_module_backward_function(sub_module, client):
    #TODO(jiaruifang) backward后处理逻辑
    logging.log(
        logging.DEBUG,
        f"After sub module backward function {sub_module.__class__.__name__} before release"
    )
    # TODO(jiaruifang) recurse
    for param in sub_module.parameters(recurse=False):
        client.release_data(param)
        client.release_grad(param)
        # FP16 grad (COMPUTE) -> FP32 grad (HOLD) on GPU, FP16 grad (FREE)
        # FP16 param (COMPUTE) -> (HOLD)


def _register_hooks_recursively(module, client, count=[0]):
    """
    DFS方式递归注册hook，father module会在children module访问后被访问一次
    但是father module的param和children module有所重复
    是否DFS只访问叶子节点比较好？
    """
    my_count = count[0]
    module.id = my_count

    logging.log(logging.DEBUG, f"{module.__class__} : {module.id}")

    for child in module.children():
        count[0] = count[0] + 1
        _register_hooks_recursively(child, client, count=count)

    # 如下两个hook和backward的hook是否重复
    def _pre_forward_module_hook(module, *args):
        pre_sub_module_forward_function(module, client)

    def _post_forward_module_hook(module, *args):
        post_sub_module_forward_function(module, client)

    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            if sub_module.applied_pre_backward is False:
                pre_sub_module_backward_function(sub_module, client)
                sub_module.applied_pre_backward = True

        return _apply_to_tensors_only(module, PreBackwardFunction,
                                      _run_before_backward_function, output)

    def _post_backward_module_hook(module, inputs):
        module.ds_grads_remaining = 0

        def _run_after_backward_function(sub_module):
            if sub_module.ds_grads_remaining == 0:
                post_sub_module_backward_function(sub_module, client)

        return _apply_to_tensors_only(module, PostBackwardFunction,
                                      _run_after_backward_function, inputs)

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)


def setup_hybrid_ps_hooks(module, client):
    _register_hooks_recursively(module, client)
