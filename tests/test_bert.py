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
import copy
from checkpoint import checkpoint
import logging
import torch
from utils import see_memory_usage
from fp16 import FP16_Module, FP16_Optimizer
import time
import argparse

from client import PatrickStarClient, PSTensorStatus, AccessType
from client import setup_hybrid_ps_hooks
from ops import CPUAdam, TorchAdam, FP16Adam
import utils.global_timer as global_timer
from runtime import Init, initialize_engine
from deepspeed_helper.global_vars import set_global_variables
from deepspeed_helper.global_vars import get_args


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
                    use_cpu_embedding: bool = False,
                    batch_size=32,
                    hidden_dim=768,
                    sequence_length=256,
                    num_layer=12,
                    stop_step=5):
    logging.info(f'test a bert model with checkpoit {is_ckp} FP16 {is_fp16}')
    logging.info(
        f'batch_size {batch_size}, hidden_dim {hidden_dim}, sequence_length {sequence_length}, num_layer {num_layer}'
    )

    args = get_args()

    # 用单卡模拟多卡
    if args.use_fake_dist:
        rank = 0
    else:
        rank = args.local_rank

    device = torch.device(f'cuda:{rank}')

    # TODO(jiaruifang) 把vocab size调小，WE层需要特殊处理？
    # rank 0在cpu上计算？
    if is_ckp:
        cfg = BertConfig(gradient_checkpointing=True,
                         hidden_dim=hidden_dim,
                         max_position_embeddings=sequence_length,
                         num_hidden_layers=num_layer)
    else:
        cfg = BertConfig(hidden_dim=hidden_dim,
                         max_position_embeddings=sequence_length,
                         num_hidden_layers=num_layer)

    if not is_ps:
        model = BertForSequenceClassification(cfg)

        model.cuda(rank)
        if is_fp16:
            model = FP16_Module(model)
        model.train()
        optimizer = TorchAdam(model.parameters(), lr=0.001)
        if is_fp16:
            optimizer = FP16_Optimizer(optimizer)

        # DPP 不能要求模型部分在cpu部分在gpu
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank])
    else:
        if is_fp16:
            client = PatrickStarClient(
                rank=rank,
                default_chunk_size=args.default_chunk_size,
                warmup=False,
                is_fp16=True)

            with Init(dtype=torch.float, client=client):
                model = BertForSequenceClassification(
                    cfg, use_cpu_embedding=args.use_cpu_embedding)

            model, optimizer, _, _ = initialize_engine(
                args=None,
                model=model,
                client=client,
                model_parameters=model.parameters())
        else:
            model = BertForSequenceClassification(cfg)
            client = PatrickStarClient(rank=rank,
                                       default_chunk_size=default_chunk_size,
                                       warmup=True,
                                       is_fp16=is_fp16)
            optimizer = CPUAdam(client, model.parameters(), lr=0.001)
            client.init(model, optimizer)

    model_numel = get_model_size(model)
    calculate_model_size(cfg)
    total_macs = calucate_MAC(cfg, batch_size, sequence_length)

    see_memory_usage(
        f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps} after model init", force=True)

    data_loader = get_bert_data_loader(
        batch_size=batch_size,
        total_samples=1000,
        sequence_length=sequence_length,
        device=device,
        data_type=torch.half if is_fp16 else torch.float,
        is_distrbuted=True)

    loss_res = []

    start_time = time.time()
    for n, batch in enumerate(data_loader):
        # if is_ps:
        #     client.pre_iter()

        step_start_time = time.time()
        output = model(input_ids=batch[0], labels=batch[1])
        loss = output.loss
        logits = output.logits
        # if torch.distributed.get_rank() == 0:
        logging.info(f"LOSS of step {n}: {loss.item()}")
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

        see_memory_usage(
            f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}  after step {n}",
            force=True)

        setp_elapse = time.time() - step_start_time
        logging.info(
            f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}  elapse {setp_elapse}, sec/iter, {total_macs/1e9/setp_elapse} GFlops"
        )
        # if is_ps:
        #     client.post_iter()
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
    #     with open('gpu_used.txt', 'w') as fh:
    #         fh.write(
    #             f'gpu_ps_used_list {len(timer.gpu_ps_used_list)} \n f{timer.gpu_ps_used_list}'
    #         )
    #         fh.write(
    #             f'gpu_used_list {len(timer.gpu_used_list)} \n {timer.gpu_used_list}'
    #         )
    #         fh.write(f'gpu_sys_used_list \n {timer.gpu_sys_used_list}')
    logging.info("*" * 20)
    return loss_res


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO)

    set_global_variables()

    args = get_args()
    use_ckp = args.use_ckp
    use_fp16 = args.use_fp16
    use_ps = args.use_ps
    # 检查结果正确性
    res_check = args.res_check
    # hidden_dim 1024, batch 16, seqence_leng 1024, ckp True.
    # PS is able to run the training, while PyTorch failed.

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo' if args.use_fake_dist else 'nccl')

    world_size = torch.distributed.get_world_size()

    plan = "A"
    if res_check:
        plan = "B"
    if plan == "A":
        # PatrickStar可以，PyTorch不可以
        # use_ckp: True, use_fp16: True, adam default on CPU, not interleave data and grad
        hidden_dim = 3072
        batch_size = 8
        sequence_length = 1024
        num_layer = 60
    elif plan == 'B':
        # PatrickStar and Torch都可以
        hidden_dim = 768
        batch_size = 1
        sequence_length = 1024
        num_layer = 3  #12
    elif plan == 'C':
        # use ckp
        # PatrickStar and PyTorch is OK
        # 没有prepare device开销
        hidden_dim = 768
        batch_size = 8
        sequence_length = 1024
        num_layer = 12
    elif plan == 'D':
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
                                    stop_step=5)
        print(loss_list)
    # calculate_mem_need(hidden_dim = hidden_dim, batch_size = batch_size, is_fp16 = use_fp16)

    if res_check:
        torch.manual_seed(0)
        loss_list = test_bert_model(is_ckp=use_ckp,
                                    is_fp16=use_fp16,
                                    is_ps=False,
                                    hidden_dim=hidden_dim,
                                    batch_size=batch_size,
                                    sequence_length=sequence_length,
                                    num_layer=num_layer)

        torch.cuda.empty_cache()
        print("*" * 50)
        torch.manual_seed(0)
        loss_ref_list = test_bert_model(is_ckp=use_ckp,
                                        is_fp16=use_fp16,
                                        is_ps=True,
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
