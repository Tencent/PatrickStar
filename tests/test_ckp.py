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

from checkpoint import checkpoint
import logging
import torch
from tests.simple_net import SimpleCKPModel, SimpleModel, get_data_loader
from patrickstar.utils import see_memory_usage
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
import time
import argparse

parser = argparse.ArgumentParser(
    description='Checkpointing for Memory Saving.')
parser.add_argument('--use_ckp',
                    dest='use_ckp',
                    action='store_true',
                    help='using checkpointing for memory saveing.')
parser.add_argument('--res_check',
                    dest='res_check',
                    action='store_true',
                    help='check results correctness of checkpointing.')
parser.add_argument('--use_fp16',
                    dest='use_fp16',
                    action='store_true',
                    help='using FP16 for training.')


def test_simple_model(is_ckp: bool = False,
                      is_fp16: bool = False,
                      hidden_dim=40,
                      batch_size=128):
    logging.info(f'test a simple model with checkpoit {is_ckp} FP16 {is_fp16}')

    device = torch.device('cuda:0')

    if is_ckp:
        model = SimpleCKPModel(hidden_dim, empty_grad=False)
    else:
        model = SimpleModel(hidden_dim, empty_grad=False)
    model.cuda()

    see_memory_usage(f"CKP {is_ckp} after model init", force=True)

    if is_fp16:
        model = FP16_Module(model)
        # model.half()

    if is_ckp:
        model.init_ckp(batch_size, torch.half if is_fp16 else torch.float)

    data_loader = get_data_loader(
        batch_size=batch_size,
        model=model,
        total_samples=1000,
        hidden_dim=hidden_dim,
        device=device,
        data_type=torch.half if is_fp16 else torch.float)

    loss_res = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if is_fp16:
        optimizer = FP16_Optimizer(optimizer)

    start_time = time.time()
    for n, batch in enumerate(data_loader):

        loss = model(batch[0], batch[1])

        # if torch.distributed.get_rank() == 0:
        print("LOSS:", loss.item())
        loss_res.append(loss.item())

        if is_fp16:
            optimizer.zero_grad(set_grads_to_None=True)
            optimizer.backward(loss, update_master_grads=False)
        else:
            optimizer.zero_grad()
            loss.backward()

        if is_fp16:
            # pass
            optimizer.update_master_grads()

        # chunk 0和 chunk 1还在compute状态
        optimizer.step()
        see_memory_usage(f"is_ckp {is_ckp} after step {n}", force=True)

        if n == 5: break

    elapse = time.time() - start_time
    logging.info(f"is_ckp {is_ckp} elapse {elapse}")
    return loss_res


def calculate_mem_need(hidden_dim, batch_size, is_fp16):
    data_size = 2 if is_fp16 else 4

    param_size = (hidden_dim * hidden_dim + hidden_dim) * 4 * data_size
    # FWD-only
    act_size = (batch_size * hidden_dim) * 4 * data_size

    # Model paramter + grad + M + V
    total_model_size = param_size * (8 if is_fp16 else 4)
    logging.info(
        f"param_size {param_size/1024} KB, total_model_size {total_model_size/1024} KB, act_size {act_size/1024} KB"
    )


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)

    args = parser.parse_args()
    use_ckp = args.use_ckp
    use_fp16 = args.use_fp16

    # 训练参数，可以自己定义
    hidden_dim = 40
    batch_size = 128
    test_simple_model(is_ckp=use_ckp,
                      is_fp16=use_fp16,
                      hidden_dim=hidden_dim,
                      batch_size=batch_size)

    calculate_mem_need(hidden_dim=hidden_dim,
                       batch_size=batch_size,
                       is_fp16=use_fp16)

    # 检查结果正确性
    res_check = args.res_check
    print(use_ckp, res_check)
    if res_check:
        torch.manual_seed(0)
        loss_ref_list = test_simple_model(is_ckp=True,
                                          is_fp16=True,
                                          hidden_dim=hidden_dim,
                                          batch_size=batch_size)

        torch.cuda.empty_cache()

        torch.manual_seed(0)
        loss_list = test_simple_model(is_ckp=False,
                                      is_fp16=True,
                                      hidden_dim=hidden_dim,
                                      batch_size=batch_size)

        print('ckp', loss_list)
        print('ref', loss_ref_list)
        for loss, loss_ref in zip(loss_list, loss_ref_list):
            assert loss == loss_ref
