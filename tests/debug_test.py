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
from torch.autograd import Variable
x = torch.Tensor([[1., 2., 3.], [4., 5., 6.]])
x = Variable(x, requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()

print(x)
print(y)
print(z)
print(out)

print(x.grad_fn)
print('y.grad_fn.next_functions[0][0]: ', y.grad_fn.next_functions[0][0])
print('z.grad_fn.next_functions[0][0]: ', z.grad_fn.next_functions[0][0])
print('out.grad_fn.next_functions[0][0]: ', out.grad_fn.next_functions[0][0])

print('y.grad_fn: ', y.grad_fn, type(y.grad_fn))
print('z.grad_fn: ', z.grad_fn)
print('out.grad_fn: ', out.grad_fn)


def reduce_ready_partitions_and_remove_grads(param, i):
    print('reduce_partition_and_remove_grads ', param)


def wrapper(param, i):
    param_tmp = param.expand_as(param)
    # if grad_cc = param_tmp.grad_fn, the hook will never been fired!
    grad_acc = param_tmp.grad_fn.next_functions[0][0]

    def reduce_partition_and_remove_grads(*notneeded):
        reduce_ready_partitions_and_remove_grads(param, i)

    grad_acc.register_hook(reduce_partition_and_remove_grads)


# y.grad_fn.next_functions[0][0].register_hook(reduce_partition_and_remove_grads)
wrapper(x, 0)
wrapper(y, 0)
wrapper(z, 0)
wrapper(out, 0)

out.backward()
