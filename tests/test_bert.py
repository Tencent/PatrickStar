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
from tests.bert_classification import BertForSequenceClassification, get_bert_data_loader
from transformers import BertConfig
import enum
import time
import sys

from checkpoint import checkpoint
import logging
import torch
from utils import see_memory_usage
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
import time
import argparse

from client import HybridPSClient
from manager import HybridPSManager
from utils import setup_hybrid_ps_hooks
from ops import CPUAdam, TorchAdam

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
parser.add_argument('--use_ps',
                    dest='use_ps',
                    action='store_true',
                    help='using Hybrid PS for training.')


def test_bert_model(is_ckp: bool = False,
                    is_fp16: bool = False,
                    is_ps: bool = False,
                    batch_size=32,
                    hidden_dim=768,
                    sequence_length=1024):
    logging.info(f'test a simple model with checkpoit {is_ckp} FP16 {is_fp16}')

    device = torch.device('cuda:0')

    if is_ckp:
        cfg = BertConfig(gradient_checkpointing=True)
    else:
        cfg = BertConfig()
    model = BertForSequenceClassification(cfg)
    model.cuda()

    see_memory_usage(
        f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps} after model init", force=True)

    if is_fp16:
        model = FP16_Module(model)

    data_loader = get_bert_data_loader(
        batch_size=batch_size,
        total_samples=1000,
        sequence_length=512,
        device=device,
        data_type=torch.half if is_fp16 else torch.float)

    loss_res = []

    if is_ps:
        manager = HybridPSManager()
        manager.init([1024 * 1024 * 1024] * 1, [1024 * 1024 * 1024 * 4 * 4])
        # chunk 32 M
        client = HybridPSClient(gpu_index=0,
                                default_chunk_size=1024 * 1024 * 32)
        optimizer = CPUAdam(client, model.parameters(), lr=0.001)
        client.register_module(model)
        setup_hybrid_ps_hooks(model, client)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if is_fp16:
        if is_ps:
            assert client is not None
        optimizer = FP16_Optimizer(optimizer)

    start_time = time.time()
    for n, batch in enumerate(data_loader):
        logging.info(
            f'input size {batch[0].shape} {batch[0].dtype}, label {batch[1].shape}'
        )
        output = model(input_ids=batch[0], labels=batch[1])
        loss = output.loss

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
        see_memory_usage(
            f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}  after step {n}",
            force=True)

        if is_ps:
            client.release_all_grad()

        if n == 5: break

    elapse = time.time() - start_time
    logging.info(f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}  elapse {elapse}")
    return loss_res


# def calculate_mem_need(hidden_dim, batch_size, is_fp16):
#     data_size = 2 if is_fp16 else 4

#     param_size = (hidden_dim * hidden_dim + hidden_dim) * 4 * data_size
#     # FWD-only
#     act_size = (batch_size * hidden_dim) * 4 * data_size

#     # Model paramter + grad + M + V
#     total_model_size = param_size * (8 if is_fp16 else 4)
#     logging.info(f"param_size {param_size/1024} KB, total_model_size {total_model_size/1024} KB, act_size {act_size/1024} KB")

if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)

    args = parser.parse_args()
    use_ckp = args.use_ckp
    use_fp16 = args.use_fp16
    use_ps = args.use_ps

    # 训练参数，可以自己定义
    hidden_dim = 768
    batch_size = 2
    sequence_length = 1024
    torch.manual_seed(0)
    test_bert_model(is_ckp=use_ckp,
                    is_fp16=use_fp16,
                    is_ps=use_ps,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    sequence_length=sequence_length)

    # calculate_mem_need(hidden_dim = hidden_dim, batch_size = batch_size, is_fp16 = use_fp16)

    # 检查结果正确性
    res_check = args.res_check

    if res_check:
        torch.manual_seed(0)
        loss_ref_list = test_bert_model(is_ckp=True,
                                        is_fp16=True,
                                        hidden_dim=hidden_dim,
                                        batch_size=batch_size)

        torch.cuda.empty_cache()

        torch.manual_seed(0)
        loss_list = test_bert_model(is_ckp=False,
                                    is_fp16=True,
                                    hidden_dim=hidden_dim,
                                    batch_size=batch_size)

        print('ckp', loss_list)
        print('ref', loss_ref_list)
        for loss, loss_ref in zip(loss_list, loss_ref_list):
            assert loss == loss_ref
