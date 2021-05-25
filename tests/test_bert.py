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
from fp16 import FP16_Module, FP16_Optimizer
import time
import argparse

from client import HybridPSClient, PSTensorStatus, AccessType
from manager import HybridPSManager
from client import setup_hybrid_ps_hooks
# from utils.zero_hook import HookedModule
from ops import CPUAdam, TorchAdam, FP16Adam
import utils.global_timer as global_timer

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


def check_grads_status(model, status):
    for name, param in model.named_parameters(recurse=True):
        param_status = param.ps_attr.get_status(AccessType.GRAD)
        assert param_status == status, f"{name} {param.ps_attr.ps_shape} {param_status} vs {status}"


def show_params(model, is_ps, step):
    print(f'show params {step}')
    for name, param in model.named_parameters(recurse=True):
        print(
            name,
            torch.sum(param) if not is_ps else torch.sum(param.ps_data_tensor),
            param.shape, param.requires_grad)


def get_model_size(model):
    numel = 0
    for name, param in model.named_parameters(recurse=True):
        numel += param.numel()
    print(f"model size {numel/1e9} B")
    return numel


def calculate_model_size(config):
    V = config.vocab_size
    N = config.num_attention_heads
    H = config.hidden_size
    L = config.num_hidden_layers
    P = config.max_position_embeddings
    numel = (V + P + (L + 1) * N + 5) * H + (L * N + 1) * (H**2)
    Embedding_numel = H * (V + P + 4)
    QKV_numel = (H * H + H) * 3
    MLP_numel = H * (4 * H) + (4 * H) + (4 * H) * H + H
    print(f"Embedding_numel layer {Embedding_numel/1e9} B")
    print(f"QKV_numel layer {QKV_numel/1e9} B")
    print(f"MLP_numel layer {MLP_numel/1e9} B")
    print(f"calcalated model size {numel/1e9} B")


def calucate_MAC(config, batch_size, sequence_length):
    B = batch_size
    S = sequence_length
    V = config.vocab_size
    N = config.num_attention_heads
    H = config.hidden_size
    L = config.num_hidden_layers
    P = config.max_position_embeddings
    cisg_total_macs = 72 * B * S * N * H**2 + 12 * B * N * H * S**2
    nvidia_total_macs = 96 * B * S * L * H**2 * (1 + S / (6 * H) + V /
                                                 (16 * L * H))

    print(f'cisg_total_macs total MACs {cisg_total_macs}')
    print(f'nvidia total MACs {nvidia_total_macs}')
    print(f'diff csig/nvidia {cisg_total_macs / nvidia_total_macs}')
    return nvidia_total_macs


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
    model_numel = get_model_size(model)
    calculate_model_size(cfg)
    total_macs = calucate_MAC(cfg, batch_size, sequence_length)

    if not is_ps:
        model.cuda()
        if is_fp16:
            model = FP16_Module(model)

    see_memory_usage(
        f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps} after model init", force=True)

    model.train()

    data_loader = get_bert_data_loader(
        batch_size=batch_size,
        total_samples=1000,
        sequence_length=sequence_length,
        device=device,
        data_type=torch.half if is_fp16 else torch.float)

    loss_res = []

    if is_ps:
        # chunk 512 MB, good for CPU-GPU bandwidth
        client = HybridPSClient(gpu_index=0,
                                default_chunk_size=1024 * 1024 * 8,
                                warmup=True,
                                is_fp16=is_fp16)

        if is_fp16:
            optimizer = FP16Adam(client,
                                 model.parameters(),
                                 lr=0.001,
                                 prefer_device=torch.device('cuda:0'))
            see_memory_usage(
                f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps} after FP16 Adam init",
                force=True)

        else:
            optimizer = CPUAdam(client,
                                model.parameters(),
                                lr=0.001,
                                prefer_device=torch.device('cuda:0'))
        client.init(model, optimizer)
    else:
        optimizer = TorchAdam(model.parameters(), lr=0.001)
        if is_fp16:
            optimizer = FP16_Optimizer(optimizer,
                                       client=client if is_ps else None)

    start_time = time.time()
    for n, batch in enumerate(data_loader):
        if is_ps:
            client.pre_iter()

        step_start_time = time.time()
        output = model(input_ids=batch[0], labels=batch[1])
        loss = output.loss
        logits = output.logits
        # if torch.distributed.get_rank() == 0:
        logging.info(f"LOSS of step {n}: {loss.item()}")
        loss_res.append(loss.item())
        if is_ps:
            timer = global_timer.IterationTimer()
            logging.info(f'FWD fininshed moment {timer.moment()}')
        if is_fp16 and not is_ps:
            optimizer.zero_grad(set_grads_to_None=True)
            optimizer.backward(loss, update_master_grads=False)
            optimizer.update_master_grads()
        else:
            optimizer.zero_grad()
            loss.backward()
            if is_ps:
                logging.info(f'BWD fininshed moment {timer.moment()}')

        # chunk 0和 chunk 1还在compute状态
        optimizer.step()
        see_memory_usage(
            f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}  after step {n}",
            force=True)

        setp_elapse = time.time() - step_start_time
        logging.info(
            f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}  elapse {setp_elapse}, sec/iter, {total_macs/1e9/setp_elapse} GFlops"
        )
        if is_ps:
            client.post_iter()
        if n == stop_step: break

    elapse = time.time() - start_time
    logging.info("*" * 20)
    logging.info(
        f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}  elapse {elapse/(stop_step+1)} sec/iter total elapse {elapse} sec"
    )
    logging.info(f"{total_macs/1e9/(elapse/(stop_step+1))} GFlops")
    logging.info(f"model numel {model_numel/1e9} B")
    if is_ps:
        global_timer.time_profiler()
        timer = global_timer.IterationTimer()
        with open('gpu_used.txt', 'w') as fh:
            fh.write(
                f'gpu_ps_used_list {len(timer.gpu_ps_used_list)} \n f{timer.gpu_ps_used_list}'
            )
            fh.write(
                f'gpu_used_list {len(timer.gpu_used_list)} \n {timer.gpu_used_list}'
            )
            fh.write(f'gpu_sys_used_list \n {timer.gpu_sys_used_list}')
    logging.info("*" * 20)
    return loss_res


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

    plan = "A"
    if res_check:
        plan = "B"
    if plan == "A":
        # HybridPS可以，PyTorch不可以
        # use_ckp: True, use_fp16: True, adam default on CPU, not interleave data and grad
        if use_fp16:
            # 精心挑选的参数
            manager = HybridPSManager()
            manager.init([1024 * 1024 * 1024 * 2],
                         [1024 * 1024 * 1024 * 4 * 4])
        else:
            manager = HybridPSManager()
            manager.init([1024 * 1024 * 512 * 4] * 2,
                         [1024 * 1024 * 1024 * 4 * 4])
        hidden_dim = 3072
        batch_size = 8
        sequence_length = 1024
        num_layer = 60
    elif plan == 'B':
        # HybridPS and Torch都可以
        manager = HybridPSManager()
        manager.init([1024 * 1024 * 1024 * 8] * 2,
                     [1024 * 1024 * 1024 * 4 * 4])
        hidden_dim = 1536
        batch_size = 8
        sequence_length = 1024
        num_layer = 12
    elif plan == 'C':
        # use ckp
        # HybridPS and PyTorch is OK
        # 没有prepare device开销
        manager = HybridPSManager()
        manager.init([1024 * 1024 * 512 * 4] * 2, [1024 * 1024 * 1024 * 4 * 4])
        hidden_dim = 768
        batch_size = 8
        sequence_length = 1024
        num_layer = 12
    elif plan == 'D':
        manager = HybridPSManager()
        manager.init([1024 * 1024 * 1024], [1024 * 1024 * 1024 * 4 * 4])
        hidden_dim = 4096  #2048
        batch_size = 2
        sequence_length = 1536
        num_layer = 120

    if not res_check:
        # 训练参数，可以自己定义
        torch.manual_seed(0)
        loss_list = test_bert_model(is_ckp=use_ckp,
                                    is_fp16=use_fp16,
                                    is_ps=use_ps,
                                    batch_size=batch_size,
                                    hidden_dim=hidden_dim,
                                    sequence_length=sequence_length,
                                    num_layer=num_layer,
                                    stop_step=3)
        print(loss_list)
    # calculate_mem_need(hidden_dim = hidden_dim, batch_size = batch_size, is_fp16 = use_fp16)

    if res_check:
        torch.manual_seed(0)
        loss_list = test_bert_model(is_ckp=use_ckp,
                                    is_fp16=use_fp16,
                                    is_ps=True,
                                    hidden_dim=hidden_dim,
                                    batch_size=batch_size,
                                    sequence_length=sequence_length,
                                    num_layer=num_layer)

        torch.cuda.empty_cache()
        print("*" * 50)
        torch.manual_seed(0)
        loss_ref_list = test_bert_model(is_ckp=use_ckp,
                                        is_fp16=use_fp16,
                                        is_ps=False,
                                        hidden_dim=hidden_dim,
                                        batch_size=batch_size,
                                        sequence_length=sequence_length,
                                        num_layer=num_layer)

        print('ps', loss_list)
        print('ref', loss_ref_list)
        import numpy as np
        print(np.array(loss_list) - np.array(loss_ref_list))
        for loss, loss_ref in zip(loss_list, loss_ref_list):
            assert loss == loss_ref
