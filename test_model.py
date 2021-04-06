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

def pre_sub_module_forward_function(sub_module):
    print(f'pre_sub_module_forward_function, access HybridPS get param')
    for param in sub_module.parameters(recurse = True):
        param.data = param.ps_tensor.data.to(param.device)


def _register_hooks_recursively(module, count=[0]):
    my_count = count[0]
    module.id = my_count

    print(f"{module.__class__} : {module.id}")

    for child in module.children():
        count[0] = count[0] + 1
        _register_hooks_recursively(child, count=count)

    def _pre_forward_module_hook(module, *args):
        pre_sub_module_forward_function(module)

    # Pre forward hook
    module.register_forward_pre_hook(_pre_forward_module_hook)


def setup_zero_stage3_hooks(module):
    _register_hooks_recursively(module)


############ UNITESTS #################

manager = HybridPSManager()

@distributed_test(world_size=1)
def test_register_module():
    world_size = dist.get_world_size()
    manager.init([32] * world_size, [64])
    print("is init manager", HybridPSManager().is_init())
    local_rank = dist.get_rank()


    hidden_dim = 4

    model = SimpleModel(hidden_dim, empty_grad=False)
    model.cuda()

    setup_zero_stage3_hooks(model)
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
    # print_params('pre-train', model)

    # client.chunk_move(0, torch.device('cuda'))
    # print_params('pre-train-move-1', model)

    # client.chunk_move(0, torch.device('cpu'))
    # print_params('pre-train-move-2', model)
    
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