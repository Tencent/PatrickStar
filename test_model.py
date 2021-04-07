import os
import json
import argparse
import torch
import deepspeed
from torch.utils.data import SequentialSampler
import torch.optim as optim
from common import distributed_test
import torch.distributed as dist

from manager import HybridPSManager
from client import HybridPSClient

class SimpleModel(torch.nn.Module):
    def __init__(self, hidden_dim, empty_grad=False):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear4 = torch.nn.Linear(hidden_dim, hidden_dim)
        if empty_grad:
            self.layers2 = torch.nn.ModuleList([torch.nn.Linear(hidden_dim, hidden_dim)])
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        hidden_dim = x
        hidden_dim = self.linear(hidden_dim)
        hidden_dim = self.linear2(hidden_dim)
        hidden_dim = self.linear3(hidden_dim)
        hidden_dim = self.linear4(hidden_dim)
        return self.cross_entropy_loss(hidden_dim, y)

def get_data_loader(model, total_samples, hidden_dim, device):
    batch_size = 4 #model.train_micro_batch_size_per_gpu()
    train_data = torch.randn(total_samples, hidden_dim, device=device, dtype=torch.float)
    train_label = torch.empty(total_samples,
                              dtype=torch.long,
                              device=device).random_(hidden_dim)
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = SequentialSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               sampler=sampler)
    return train_loader

def print0(msg):
    print(msg, flush=True)


def print_params(tag, model):
    # if torch.distributed.get_rank() == 0:
    for n, p in model.named_parameters():
        print0(f"tag: {tag}, n: {n}, p: {p}, p: {p.device}, grad: {p.grad}")
        print0(f"tag: {tag}, n: {n}, p.ps_tensor: {p.ps_tensor}, {p.ps_tensor.device}")


############# HOOKS ####################

# 可以修改outputs
class PreBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module, pre_backward_function, outputs):
        ctx.module = module
        ctx.pre_backward_function = pre_backward_function
        module.applied_pre_backward = False
        print(f"After Forward: {ctx.module.__class__.__name__}")
        # why detach?detach后给下一层作为输入，似乎没有用，fwd输出都会用backward作为反向
        outputs = outputs.detach()
        return outputs

    @staticmethod
    def backward(ctx, *args):
        print(f"Before Backward: {ctx.module.__class__.__name__}")
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
            print(f"After Backward: {ctx.module.__class__.__name__}")
        # why (None, None) as first two returns
        else:
            print(f"After Backward: {ctx.module.__class__.__name__} None, None")
        print('output args', args)
        return (None, None) + args

#apply torch.autograd.Function that calls a backward_function to tensors in output
def _apply_to_tensors_only(module, functional, backward_function, outputs):
    if type(outputs) is tuple:
        touched_outputs = []
        for output in outputs:
            touched_output = _apply_to_tensors_only(module,
                                                    functional,
                                                    backward_function,
                                                    output)
            touched_outputs.append(touched_output)
        return tuple(touched_outputs)
    elif type(outputs) is torch.Tensor:
        return functional.apply(module, backward_function, outputs)
    else:
        return outputs

# 必须具备重复调用，第二次无效的能力 fetch submodule
def pre_sub_module_forward_function(sub_module, client):
    print(f'{sub_module.__class__} pre_sub_module_forward_function, access HybridPS get param')
    for param in sub_module.parameters(recurse = True):
        # TODO(jiaruifang)以chunk方式访问
        # param.data = param.ps_tensor.data.to(param.device)
        # print(f'param is originally on {param.original_device}, ps_tensor on {param.ps_tensor.device}, {param.data.data_ptr()}')
        client.access(param)
        param.data = param.ps_tensor.data
        # print(f'param is now on {param.data.device} {param.data.data_ptr()}')
        # pass

# release submodule
def post_sub_module_forward_function(sub_module, client):
    print(f'{sub_module.__class__} post_sub_module_forward_function, access HybridPS get param')
    for param in sub_module.parameters(recurse = True):
        client.release(param)

def pre_sub_module_backward_function(sub_module, client):
    # TODO(jiaruifang) backward前处理逻辑
    print(f'Before sub module backward function {sub_module.__class__.__name__} allgather')
    for param in sub_module.parameters(recurse = True):
        client.access(param)
        param.data = param.ps_tensor.data
    
# release param of submodule
def post_sub_module_backward_function(sub_module):
    #TODO(jiaruifang) backward后处理逻辑
    print(
        f"After sub module backward function {sub_module.__class__.__name__} before release")
        
def _register_hooks_recursively(module, client, count=[0]):
    """
    DFS方式递归注册hook，father module会在children module访问后被访问一次
    但是father module的param和children module有所重复
    是否DFS只访问叶子节点比较好？
    """
    my_count = count[0]
    module.id = my_count

    print(f"{module.__class__} : {module.id}")

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

        return _apply_to_tensors_only(module,
                                        PreBackwardFunction,
                                        _run_before_backward_function,
                                        output)

    def _post_backward_module_hook(module, inputs):
        module.ds_grads_remaining = 0

        def _run_after_backward_function(sub_module):
            if sub_module.ds_grads_remaining == 0:
                post_sub_module_backward_function(sub_module)

        return _apply_to_tensors_only(module,
                                        PostBackwardFunction,
                                        _run_after_backward_function,
                                        inputs)
                                        
    module.register_forward_pre_hook(_pre_forward_module_hook)
    module.register_forward_hook(_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)

def setup_zero_stage3_hooks(module, client):
    _register_hooks_recursively(module, client)


############ UNITESTS #################

manager = HybridPSManager()

@distributed_test(world_size=1)
def test_register_module():
    world_size = dist.get_world_size()
    # 测试用例中GPU显存32，CPU内存64
    manager.init([32] * world_size, [64])
    print("is init manager", HybridPSManager().is_init())
    local_rank = dist.get_rank()


    hidden_dim = 4

    model = SimpleModel(hidden_dim, empty_grad=False)
    model.cuda()

    # param's grad is None
    # 将param和grad用一块自定义的存储空间接管

    data_loader = get_data_loader(model=model,
                              total_samples=1000,
                              hidden_dim=hidden_dim,
                              device=torch.cuda.current_device())

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    client = HybridPSClient(gpu_index = local_rank, 
                          default_chunk_size = 20)
    
    client.register_module(model)
    setup_zero_stage3_hooks(model, client)
    # print_params('pre-train', model)

    # client.chunk_move(0, torch.device('cuda'))
    # print_params('pre-train-move-1', model)

    # client.chunk_move(0, torch.device('cpu'))
    # print_params('pre-train-move-2', model)
    
    for n, p in model.named_parameters():
        assert p.original_device.type == 'cuda'

    for n, batch in enumerate(data_loader):
        optimizer.zero_grad()
        loss = model(batch[0], batch[1])
        # if torch.distributed.get_rank() == 0:
        print("LOSS:", loss.item())
        # model.backward(loss)
        loss.backward()
        # model.step()
        optimizer.step()
        print_params('step={}'.format(n), model)
        if n == 5: break

test_register_module()