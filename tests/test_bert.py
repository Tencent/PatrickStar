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
import logging
import time
import argparse

from patrickstar.utils import see_memory_usage
from patrickstar.fp16 import FP16_Module, FP16_Optimizer
from patrickstar.core import PatrickStarClient
from patrickstar.ops import CPUAdam, TorchAdam, FP16Adam
import patrickstar.utils.global_timer as global_timer
from patrickstar.runtime import Init, initialize_engine
from patrickstar.deepspeed_helper.global_vars import set_global_variables
from patrickstar.deepspeed_helper.global_vars import get_args
from patrickstar.manager import PatrickStarManager
from patrickstar.utils.model_size_calculator import get_ps_model_size, estimate_bert_MAC


def show_params(model, is_ps, step):
    print(f'show params {step}')
    for name, param in model.named_parameters(recurse=True):
        print(
            name,
            torch.sum(param) if not is_ps else torch.sum(param.ps_data_tensor),
            param.shape, param.requires_grad)


def test_bert_model(is_ckp: bool = False,
                    is_fp16: bool = False,
                    is_ps: bool = False,
                    use_cpu_embedding: bool = False,
                    batch_size=32,
                    hidden_dim=768,
                    sequence_length=256,
                    num_layer=12,
                    num_head=12,
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
                         hidden_size=hidden_dim,
                         intermediate_size=hidden_dim * 4,
                         num_attention_heads=num_head,
                         max_position_embeddings=sequence_length,
                         num_hidden_layers=num_layer)
    else:
        cfg = BertConfig(hidden_size=hidden_dim,
                         intermediate_size=hidden_dim * 4,
                         max_position_embeddings=sequence_length,
                         num_attention_heads=num_head,
                         num_hidden_layers=num_layer)

    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-6
    weight_decay = 0

    if not is_ps:
        model = BertForSequenceClassification(cfg)

        model.cuda(rank)
        if is_fp16:
            model = FP16_Module(model)
        model.train()
        optimizer = TorchAdam(model.parameters(),
                              lr=lr,
                              betas=betas,
                              eps=eps,
                              weight_decay=weight_decay)
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
                warmup=True,
                is_fp16=True)

            with Init(dtype=torch.float, client=client):
                model = BertForSequenceClassification(
                    cfg, use_cpu_embedding=args.use_cpu_embedding)

            model, optimizer, _, _ = initialize_engine(
                args=None,
                model=model,
                client=client,
                model_parameters=model.parameters(),
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay)
        else:
            model = BertForSequenceClassification(cfg)
            client = PatrickStarClient(
                rank=rank,
                default_chunk_size=args.default_chunk_size,
                warmup=True,
                is_fp16=is_fp16)
            optimizer = CPUAdam(client, model.parameters(), lr=lr)
            client.init(model, optimizer)

    model_numel = get_ps_model_size(model)
    total_macs = estimate_bert_MAC(cfg, batch_size, sequence_length)

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

    # 开启预热优化
    if is_ps:
        mgr = PatrickStarManager()
        mgr.start_train(
            is_warmup=True,
            param_fp16_chunk_size=client.get_param_fp16_chunks_mem_size(),
            chunk_size=args.default_chunk_size)

    # logging.info(f"{total_macs/1e9/(elapse/(stop_step+1))} GFlops")
    logging.info(f"model numel {model_numel/1e9} B")

    for n, batch in enumerate(data_loader):
        step_start_time = time.time()
        output = model(input_ids=batch[0], labels=batch[1])
        loss = output.loss
        logits = output.logits
        # if torch.distributed.get_rank() == 0:
        logging.info(f"LOSS of step {n}: {loss.item()}")
        loss_res.append(loss.item())

        # logging.info(f'FWD finished moment {timer.moment()}')
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

        # logging.info(f'BWD finished moment {timer.moment()}')
        optimizer.step()

        see_memory_usage(
            f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}  after step {n}",
            force=True)

        step_elapse = time.time() - step_start_time
        logging.info(
            f"ckp {is_ckp} fp16 {is_fp16} ps {is_ps}: step elapse {step_elapse} sec/iter, {total_macs/1e9/step_elapse} GFlops"
        )

        if is_ps:
            global_timer.my_timer.print()
            global_timer.my_timer.reset()

            global_timer.data_move_cnter.print()
            global_timer.data_move_cnter.reset()
        if n == stop_step: break

    elapse = time.time() - start_time

    if is_ps:
        mgr = PatrickStarManager()
        mgr.show_mem_curve()

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

    plan = "GPT3mid"
    if res_check:
        plan = "GPTsmall"
    if plan == "GPTsmall":
        # 0.11B
        hidden_dim = 768
        batch_size = 128
        sequence_length = 128
        num_layer = 6
        num_head = 12
    elif plan == 'GPT3mid':
        # 0.35B
        # PatrickStar and Torch都可以
        hidden_dim = 1024
        batch_size = 128
        sequence_length = 128
        num_layer = 24
        num_head = 16
    elif plan == 'GPT3large':
        # 0.7B
        hidden_dim = 1536
        batch_size = 64
        sequence_length = 128
        num_layer = 24
        num_head = 16
    elif plan == 'GPT3_1B':
        # 0.9B
        hidden_dim = 1536
        batch_size = 64
        sequence_length = 128
        num_layer = 30
        num_head = 16
    elif plan == 'GPT3larger':
        # 1.27B
        hidden_dim = 2048
        batch_size = 8
        sequence_length = 128
        num_layer = 24
        num_head = 16
    elif plan == 'GPT3XL':
        # 1.3B model
        hidden_dim = 2048
        batch_size = 8
        sequence_length = 512
        num_layer = 24
        num_head = 32
    elif plan == 'GPT3_2B':
        # 2.7B model
        hidden_dim = 2560  #2048
        batch_size = 8
        sequence_length = 128
        num_layer = 32
        num_head = 32
    elif plan == 'GPT3_6B':
        # 6.7B model
        hidden_dim = 4096
        batch_size = 8
        sequence_length = 512
        num_layer = 32
        num_head = 32
    elif plan == 'GPT3_10B':
        # 13B model
        hidden_dim = 5140
        batch_size = 8
        sequence_length = 512
        num_layer = 32
        num_head = 40
    if res_check:
        batch_size = 2

    assert hidden_dim % num_head == 0
    logging.info(f'Benchmarking {plan}')
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
                                    num_head=num_head,
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
                                    num_layer=num_layer,
                                    num_head=num_head)

        torch.cuda.empty_cache()
        print("*" * 50)
        torch.manual_seed(0)
        loss_ref_list = test_bert_model(is_ckp=use_ckp,
                                        is_fp16=use_fp16,
                                        is_ps=True,
                                        hidden_dim=hidden_dim,
                                        batch_size=batch_size,
                                        sequence_length=sequence_length,
                                        num_layer=num_layer,
                                        num_head=num_head)

        print('ps', loss_list)
        print('ref', loss_ref_list)
        import numpy as np
        print(np.array(loss_list) - np.array(loss_ref_list))
        for loss, loss_ref in zip(loss_list, loss_ref_list):
            assert loss == loss_ref
