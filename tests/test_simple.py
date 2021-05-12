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

import os
import json
import argparse
import torch
from torch.utils.data import SequentialSampler
import torch.optim as optim
import logging
import time

from ops import CPUAdam, TorchAdam
from client import HybridPSClient, setup_hybrid_ps_hooks, PSTensorStatus
from manager import HybridPSManager
from utils import see_memory_usage
import utils.global_timer as global_timer

from fp16 import configure_fp16_optimizer
from fp16 import FP16_Module
from fp16 import FP16_Optimizer

from tests.simple_net import SimpleModel, get_data_loader


def show_optim(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            print(p.size())


def test_simple_model(is_ps: bool = False, is_fp16: bool = False):
    logging.info(f'test a simple model with hybrid ps {is_ps} FP16 {is_fp16}')

    hidden_dim = 4
    batch_size = 4
    device = torch.device('cuda:0')

    model = SimpleModel(hidden_dim, empty_grad=False)
    model.cuda()

    see_memory_usage(f"PS {is_ps} after model init", force=True)

    if is_fp16:
        model = FP16_Module(model)
        # model.half()

    data_loader = get_data_loader(
        batch_size=batch_size,
        total_samples=1000,
        hidden_dim=hidden_dim,
        device=device,
        data_type=torch.half if is_fp16 else torch.float)

    loss_res = []
    if is_ps:
        client = HybridPSClient(gpu_index=0, default_chunk_size=20)
        optimizer = CPUAdam(client, model.parameters(), lr=0.001)
    else:
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer = TorchAdam(model.parameters(), lr=0.001)

    if is_fp16:
        if is_ps:
            assert (client is not None)
        optimizer = FP16_Optimizer(optimizer, client=client if is_ps else None)
        # optimizer = configure_fp16_optimizer(optimizer)

    if is_ps:
        client.register_model_optimizer(model, optimizer)

    start_time = time.time()
    for n, batch in enumerate(data_loader):

        if n == 0:
            global_timer.record_chunk_lifecycle = True

        loss = model(batch[0], batch[1])

        # if torch.distributed.get_rank() == 0:
        print("LOSS:", loss.item())
        loss_res.append(loss.item())

        if is_fp16:
            optimizer.zero_grad(set_grads_to_None=True)
            optimizer.backward(loss, update_master_grads=False)
            if is_ps:
                client.release_all_data_grad(PSTensorStatus.HOLD)
        else:
            optimizer.zero_grad()
            loss.backward()
            if is_ps:
                client.release_all_data_grad(PSTensorStatus.HOLD)

        if is_fp16:
            # pass
            optimizer.update_master_grads()

        # chunk 0和 chunk 1还在compute状态
        optimizer.step()
        see_memory_usage(f"PS {is_ps} after step {n}", force=True)

        # it is necessary to get correct results
        if is_ps:
            client.release_all_data_grad(PSTensorStatus.HOLD)

        if n == 0:
            global_timer.record_chunk_lifecycle = False

        if n == 5: break

    elapse = time.time() - start_time
    logging.info(f"is_ps {is_ps} elapse {elapse}")
    logging.info("======================" * 4)
    if is_ps:
        client.chunk_list.visit()
        global_timer.time_profiler()

    return loss_res


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)
    torch.manual_seed(0)
    manager = HybridPSManager()
    # 4 layer每层20个elem(20*4 bytes)，最少360 (360*4 bytes)内存
    # gpu内存至少为40，反向传播一层需要的最大内存。

    test_cpu_adam = False
    if test_cpu_adam:
        # manager.init([40 * 4] * 1, [280 * 4])
        manager.init([180 * 4] * 1, [280 * 4])
        loss_ref_list = test_simple_model(False)

        torch.manual_seed(0)
        loss_list = test_simple_model(True)

        print('hybridps', loss_list)
        print('ref', loss_ref_list)
        for loss, loss_ref in zip(loss_list, loss_ref_list):
            assert loss == loss_ref

        # print(loss_list)
        # print('gpu usage ', manager.gpu_mem_usage_curve)
        # print('cpu usgae ', manager.cpu_mem_usage_curve)

    test_fp16 = True

    if test_fp16:
        # TODO(jiaruifang) 内存释放干净
        # M, V, G32, P32 = 360
        # P16 = 80/2=40
        # G16 = 80/2=40
        manager.reset([40000 * 4] * 1, [32000 * 4 + 2 * 2000])
        torch.manual_seed(0)
        loss_list = test_simple_model(True, is_fp16=True)
        see_memory_usage("after HybridPS simple model", force=True)

        torch.manual_seed(0)
        loss_list_ref = test_simple_model(False, is_fp16=True)

        print('ps loss', loss_list)
        print('ref loss', loss_list_ref)

        for loss, loss_ref in zip(loss_list, loss_list_ref):
            assert loss == loss_ref
