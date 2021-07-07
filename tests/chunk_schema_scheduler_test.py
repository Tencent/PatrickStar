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
from patrickstar.utils import see_memory_usage
from fp16 import FP16_Module
from fp16 import FP16_Optimizer
import time
import argparse

from client import PatrickStarClient, PSTensorStatus
from manager import PatrickStarManager
from client import setup_hybrid_ps_hooks, ChunkShemaScheduler
# from patrickstar.utils.zero_hook import HookedModule
from ops import CPUAdam, TorchAdam
import patrickstar.utils.global_timer as global_timer

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
                    sequence_length=256,
                    num_layer=12,
                    stop_step=10):
    logging.info(f'test a simple model with checkpoit {is_ckp} FP16 {is_fp16}')
    logging.info(
        f'batch_size {batch_size}, hidden_dim {hidden_dim}, sequence_length {sequence_length}, num_layer {num_layer}'
    )

    device = torch.device('cuda:0')

    if is_ckp:
        cfg = BertConfig(gradient_checkpointing=True,
                         hidden_dim=hidden_dim,
                         max_position_embeddings=sequence_length,
                         num_hidden_layers=num_layer)
    else:
        cfg = BertConfig(hidden_dim=hidden_dim,
                         max_position_embeddings=sequence_length,
                         num_hidden_layers=num_layer)
    model = BertForSequenceClassification(cfg)
    model.cuda()
    model.train()

    see_memory_usage(
        f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps} after model init", force=True)

    if is_fp16:
        model = FP16_Module(model)

    manager = PatrickStarManager()
    manager.init([1024 * 1024 * 1024 * 2] * 1, [1024 * 1024 * 1024 * 4 * 4])
    # chunk 512 MB, good for CPU-GPU bandwidth
    client = PatrickStarClient(rank=0, default_chunk_size=1024 * 1024)

    optimizer = CPUAdam(client, model.parameters(), lr=0.001)
    if is_fp16:
        optimizer = FP16_Optimizer(optimizer, client=None, verbose=True)
    client.register_model_optimizer(model, optimizer)


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
    # 检查结果正确性
    res_check = args.res_check

    # hidden_dim 1024, batch 16, seqence_leng 1024, ckp True.
    # PS is able to run the training, while PyTorch failed.

    plan = "C"
    if plan == "A":
        hidden_dim = 3072  #2048
        batch_size = 16
        sequence_length = 1024
        num_layer = 60
    elif plan == 'B':
        # PatrickStar and Pytorch都可以
        # Pytorch: 1.2852387428283691 sec
        # PatrickStar: 6.879993915557861 sec
        # client_prepare_device_elapse 0.0 client_access_elapse 2.211916446685791 client_release_elapse 2.442206859588623
        # cpu_adam_elapse 3.7840416431427 cpu_adam_f_elapse 3.7840394973754883
        hidden_dim = 1536
        batch_size = 8
        sequence_length = 1024
        num_layer = 12
    elif plan == 'C':
        # use ckp
        # PatrickStar and PyTorch is OK
        # 没有prepare device开销
        hidden_dim = 768
        batch_size = 8
        sequence_length = 1024
        num_layer = 1
    elif plan == 'D':
        # use ckp
        # PatrickStar and PyTorch is OK
        # 没有prepare device开销
        hidden_dim = 4096  #2048
        batch_size = 2
        sequence_length = 1536
        num_layer = 120

    torch.manual_seed(0)
    test_bert_model(is_ckp=use_ckp,
                    is_fp16=use_fp16,
                    is_ps=use_ps,
                    batch_size=batch_size,
                    hidden_dim=hidden_dim,
                    sequence_length=sequence_length,
                    num_layer=num_layer)
