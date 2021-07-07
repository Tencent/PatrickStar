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

from ops import CPUAdam, TorchAdam, FP16Adam
from client import PatrickStarClient, setup_hybrid_ps_hooks, PSTensorStatus
from manager import PatrickStarManager
from patrickstar.utils import see_memory_usage
import patrickstar.utils.global_timer as global_timer

from fp16 import configure_fp16_optimizer
from fp16 import FP16_Module
from fp16 import FP16_Optimizer

from tests.simple_net import SimpleModel, get_data_loader, get_bert_data_loader
from runtime import initialize_engine, Init
from deepspeed_helper.global_vars import set_global_variables
from deepspeed_helper.global_vars import get_args


def show_optim(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            print(p.size())


def test_simple_model(is_ps: bool = False,
                      is_fp16: bool = False,
                      is_ckp: bool = True,
                      use_cpu_embedding: bool = False,
                      stop_iter: int = 10):
    logging.info(f'test a simple model with hybrid ps {is_ps} FP16 {is_fp16}')
    args = get_args()
    hidden_dim = 4
    batch_size = 4

    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        torch.distributed.init_process_group(backend='nccl')
        rank = args.local_rank
    device = torch.device(f'cuda:{rank}')

    if not is_ps:
        if use_cpu_embedding:
            model = SimpleModel(hidden_dim,
                                is_ckp=is_ckp,
                                use_cpu_embedding=True)
            model.encoder.cuda(rank)
        else:
            model = SimpleModel(hidden_dim,
                                is_ckp=is_ckp,
                                use_cpu_embedding=False)
            model.cuda(rank)
        if is_fp16:
            model = FP16_Module(model)
        model.train()
        optimizer = TorchAdam(model.parameters(), lr=0.001)
        if is_fp16:
            optimizer = FP16_Optimizer(optimizer)
    else:
        if is_fp16:
            client = PatrickStarClient(
                rank=rank,
                default_chunk_size=args.default_chunk_size,
                warmup=True,
                is_fp16=True)

            with Init(dtype=torch.float):
                model = SimpleModel(hidden_dim,
                                    is_ckp=is_ckp,
                                    use_cpu_embedding=args.use_cpu_embedding)

            model, optimizer, _, _ = initialize_engine(
                args=None,
                model=model,
                client=client,
                model_parameters=model.parameters())
        else:
            model = SimpleModel(hidden_dim, is_ckp=is_ckp)
            client = PatrickStarClient(rank=0,
                                       default_chunk_size=20,
                                       warmup=True,
                                       is_fp16=is_fp16)
            optimizer = CPUAdam(client, model.parameters(), lr=0.001)
            client.init(model, optimizer)

    see_memory_usage(f"PS {is_ps} after model init", force=True)

    data_loader = get_bert_data_loader(batch_size=batch_size,
                                       total_samples=100000,
                                       sequence_length=10,
                                       device=device,
                                       is_distrbuted=True)

    loss_res = []

    start_time = time.time()

    if is_ps:
        mgr = PatrickStarManager()
        mgr.start_train(is_warmup=True)

    for n, batch in enumerate(data_loader):
        loss = model(batch[0], batch[1])

        # if torch.distributed.get_rank() == 0:
        print(f"LOSS: {loss.item()} at {n}")
        loss_res.append(loss.item())

        if not is_ps:
            if is_fp16:
                optimizer.zero_grad(set_grads_to_None=True)
                optimizer.backward(loss, update_master_grads=False)
                optimizer.update_master_grads()
            else:
                optimizer.zero_grad()
                loss.backward()
        else:
            if is_fp16:
                model.backward(loss)
            else:
                optimizer.zero_grad()
                loss.backward()
        optimizer.step()

        see_memory_usage(f"PS {is_ps} after step {n}", force=True)

        if is_ps:
            global_timer.my_timer.print()
            global_timer.my_timer.reset()

        if n == stop_iter: break

    elapse = time.time() - start_time
    logging.info(f"is_ps {is_ps} elapse {elapse}")
    logging.info("======================" * 4)

    if is_ps:
        # client.chunk_list.visit()
        global_timer.time_profiler()

    return loss_res


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)
    set_global_variables()
    torch.manual_seed(0)
    manager = PatrickStarManager()
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

    test_fp16 = True
    if test_fp16:
        # hidden_dim = 4
        # 需要 40和8两个chunk
        manager.reset([48 * 2 * 4] * 1, [160 * 4 * 2 + 2 * 160])
        torch.manual_seed(0)
        loss_list = test_simple_model(is_ps=False,
                                      is_fp16=False,
                                      is_ckp=True,
                                      use_cpu_embedding=False)
        # see_memory_usage("after PatrickStar simple model", force=True)

        torch.manual_seed(0)
        loss_list_ref = test_simple_model(is_ps=False,
                                          is_fp16=False,
                                          is_ckp=True,
                                          use_cpu_embedding=True)

        print('ps loss', loss_list)
        print('ref loss', loss_list_ref)

        for loss, loss_ref in zip(loss_list, loss_list_ref):
            assert loss == loss_ref
