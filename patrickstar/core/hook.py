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
from .const import PSTensorStatus, AccessType, TrainingStage
import patrickstar.utils.global_timer as global_timer
from patrickstar.utils import logger
from patrickstar.core.parameter import ParamType
from patrickstar.manager import PatrickStarManager


############# HOOKS ####################
#for each tensor in outputs run the forward_funciton and register backward_function as hook
def _apply_forward_and_backward_to_tensors_only(module, forward_function,
                                                backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_forward_and_backward_to_tensors_only(
                module, forward_function, backward_function, output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        forward_function(outputs)
        if outputs.requires_grad:
            logging.debug('output require grad {outputs.shape}')
            outputs.register_hook(backward_function)
        else:
            logging.debug('output dose not require grad {outputs}')
        return outputs
    else:
        return outputs


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
        # logging.debug(f'_apply_to_tensors_only {module}')
        return functional.apply(module, backward_function, outputs)
    else:
        # print('_apply_to_tensors_only', outputs)
        return outputs


# 可以修改outputs
class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        logging.log(logging.DEBUG,
                    f"**After Forward: {ctx.module.__class__.__name__}")
        # why detach?detach后给下一层作为输入，似乎没有用，fwd输出都会用backward作为反向
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        logging.log(logging.DEBUG,
                    f"**Before Backward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


class PostBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, output):
        ctx.module = module
        # if output.requires_grad:
        #     #TODO SOME TIMES post backward does not seem to be triggered debug in detail
        #     #Should only cause increase in memory not correctness issue
        #     #if output.grad_fn.__class__.__name__ == 'ViewBackward':
        #     #    ctx.view=True
        #     #    print(f"Warning view tensor for input to module : {module.__class__.__name__}. Backward hooks may not trigger properly")
        #     #assert len(module.parameters(recurse=False)), "The input tensor to the module is a view, and autograd Function or register_hook is not triggered with view tensors."
        #     #if module.ds_grads_remaining == 0:
        #     #    print(f"Before Forward: {ctx.module.__class__.__name__}")
        #     # module.ds_grads_remaining += 1
        #     ctx.pre_backward_function = pre_backward_function
        output = output.detach()
        # output = output * 1
        logging.debug(
            f"**PostBackwardFunction forward: {ctx.module.__class__.__name__}")
        ctx.pre_backward_function = pre_backward_function
        return output

    # arguments: 下一层的activation_grad, rets: 输入activation的grad
    @staticmethod
    def backward(ctx, *args):
        # ctx.module.ds_grads_remaining = ctx.module.ds_grads_remaining - 1
        # if ctx.module.ds_grads_remaining == 0:
        # 没有执行embedding
        logging.log(
            logging.DEBUG,
            f"**PostBackwardFunction backward: {ctx.module.__class__.__name__}"
        )
        ctx.pre_backward_function(ctx.module)
        return (None, None) + args


# 必须具备重复调用，第二次无效的能力 fetch submodule
def pre_sub_module_forward_function(sub_module, client, name):
    logger.debug(
        f'pre_sub_module_forward_function {sub_module.__class__.__name__}')
    flag = False
    for sub_name, param in sub_module.named_parameters(recurse=False):
        rank = torch.distributed.get_rank()
        logger.debug(f'rank {rank} FWD pre {name}.{sub_name} access data')
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        param.data = client.access_dist(
            param,
            AccessType.DATA,
            torch.device(f'cuda:{client.local_rank}'),
            training_stage=TrainingStage.FWD)
        flag = True
    if flag:
        mgr = PatrickStarManager()
        mgr.tiktac(client)


# release submodule
def post_sub_module_forward_function(sub_module, client, name):
    for sub_name, param in sub_module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        rank = torch.distributed.get_rank()
        logger.debug(f'rank {rank} FWD post {name}.{sub_name}')
        client.release_dist(param,
                            AccessType.DATA,
                            PSTensorStatus.HOLD_AFTER_FWD,
                            training_stage=TrainingStage.FWD,
                            is_allreduce=False)
    mgr = PatrickStarManager()
    mgr.tiktac(client)


def pre_sub_module_backward_function(sub_module, client, name):
    flag = False
    for sub_name, param in sub_module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        rank = torch.distributed.get_rank()
        logger.debug(f'rank {rank} BWD pre {name}.{sub_name}')
        if param.ps_attr.data_type == torch.half:
            rank = torch.distributed.get_rank()
            tmp_tensor = client.access_dist(
                param,
                AccessType.DATA,
                torch.device(f'cuda:{client.local_rank}'),
                training_stage=TrainingStage.BWD)
            param.data = tmp_tensor
            param.grad = torch.zeros_like(tmp_tensor)
            assert param.data.data_ptr() != param.grad.data_ptr()

            assert param.ps_attr.bwd_cnt == 0, f"Backward Propagation updates the gradient of a parameter twice. This is not allowed when using chunk reusing."

        elif param.ps_attr.data_type == torch.float:
            raise RuntimeError("fp32 training is not supported!")
        flag = True
    if flag:
        mgr = PatrickStarManager()
        mgr.tiktac(client)


def post_sub_module_backward_function(sub_module, client, name):
    for sub_name, param in sub_module.named_parameters(recurse=False):
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            continue
        # TODO(jiaruifang)以后给所有param的ps_attr加一个FP16或者FP32的标识。
        # 不要用param的本身类型判断。因为这个我们可以随意更改param，并没有加检测或者权限限制。
        if param.ps_attr.data_type == torch.half:
            client.optimizer.check_overflow(param)
            tmp_tensor = param.ps_attr.access_tensor(AccessType.DATA)
            tmp_tensor.copy_(param.grad)
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                # 正确的梯度
                logger.debug(
                    f'rank {rank} BWD post before release_dist {name}.{sub_name}'
                )
                client.release_dist(param,
                                    AccessType.DATA,
                                    PSTensorStatus.HOLD_AFTER_BWD,
                                    training_stage=TrainingStage.BWD,
                                    is_allreduce=True)

            else:
                client.release_data(param, PSTensorStatus.HOLD)

            param.grad = None
            param.ps_attr.bwd_cnt += 1
        elif param.ps_attr.data_type == torch.float:
            client.release_grad(param, PSTensorStatus.HOLD)
            client.release_data(param, PSTensorStatus.HOLD)
    mgr = PatrickStarManager()
    mgr.tiktac(client)


def _register_hooks_recursively(module, client, count=[0], name=""):
    """
    DFS方式递归注册hook，father module会在children module访问后被访问一次
    但是father module的param和children module有所重复
    是否DFS只访问叶子节点比较好？
    """
    my_count = count[0]
    module.id = my_count

    has_children = False
    for child_name, child in module.named_children():
        count[0] = count[0] + 1
        _register_hooks_recursively(child, client, count, name + child_name)
        has_children = True
    if has_children:
        return

    # 如下两个hook和backward的hook是否重复
    def _pre_forward_module_hook(module, *args):
        pre_sub_module_forward_function(module, client, name)

    def _post_forward_module_hook(module, *args):
        post_sub_module_forward_function(module, client, name)

    # pre_bwd_hook_times = 0
    # post_bwd_hook_times = 0
    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            # if sub_module.applied_pre_backward is False:
            pre_sub_module_backward_function(sub_module, client, name)
            # sub_module.applied_pre_backward = True

        # self.pre_bwd_hook_times += len(output)
        return _apply_to_tensors_only(module, PreBackwardFunction,
                                      _run_before_backward_function, output)

    def _post_backward_module_hook(module, inputs):
        # module.ds_grads_remaining = 0
        for input in inputs:
            if input is not None:
                if not isinstance(input, torch.Tensor):
                    logging.debug(f'_post_backward_module_hook {input}')
                else:
                    logging.debug(
                        f'_post_backward_module_hook {module.id} {input.shape}'
                    )

        def _run_after_backward_function(sub_module):
            # if sub_module.ds_grads_remaining == 0:
            post_sub_module_backward_function(sub_module, client, name)

        # self.post_bwd_hook_times += len(inputs)
        return _apply_to_tensors_only(module, PostBackwardFunction,
                                      _run_after_backward_function, inputs)

    logger.debug(f"_register_hooks_recursively {module.__class__.__name__}")

    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)


def setup_patrickstar_hooks(module, client):
    _register_hooks_recursively(module, client)

    def make_post_backward_hook(param):
        def hook(*ignore):
            global_timer.my_timer.start_profile('HOOK_torch_allreduce')
            world_size = torch.distributed.get_world_size()
            client.optimizer.check_overflow(param)
            torch.distributed.all_reduce(param.grad,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=client.cpu_comm_group,
                                         async_op=False)
            param.data /= world_size
            logger.debug(
                f'rank {torch.distributed.get_rank} allreduce grad {param.ps_attr.name}'
            )
            global_timer.my_timer.finish_profile('HOOK_torch_allreduce')

        return hook

    # torch param will not override data with grad,
    # we could use the standard register_hook on them.
    for param in client.torch_param_list:
        if param.requires_grad:
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(make_post_backward_hook(param))
            client.grad_accs.append(grad_acc)
