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
from data_loader import get_bert_data_loader
from transformers import BertConfig, BertForSequenceClassification
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
from patrickstar.ops import FP16Adam
import patrickstar.utils.global_timer as global_timer
from patrickstar.runtime import initialize_engine
from patrickstar.manager import PatrickStarManager
from patrickstar.utils.model_size_calculator import get_ps_model_size, estimate_bert_MAC
import os
from patrickstar.utils.logging import logger


def _add_patrick_star_args(parser):
    group = parser.add_argument_group(title='partickstar')
    group.add_argument('--use_fake_dist',
                       dest='use_fake_dist',
                       action='store_true',
                       help='using one GPU to stimulate multiple card.')
    group.add_argument('--batch_size',
                       type=int,
                       default=32,
                       help='Batch size of input.')
    group.add_argument('--default_chunk_size',
                       type=int,
                       default=32 * 1024 * 1024,
                       help='Default Chunk Size in elements.')

    group.add_argument(
        '--use_cpu_embedding',
        dest='use_cpu_embedding',
        action='store_true',
        help=
        'using CPU to perform Embedding and do not assign embedding params to chunks'
    )
    group.add_argument('--use_deepspeed_cpu_adam',
                       action='store_true',
                       help='Use deepspeed cpu adam')
    group.add_argument(
        '--use_hybrid_adam',
        action='store_true',
        help=
        'Use hybrid adam optimization. By default ADAM is on CPU and run ADAM on GPU if possible.'
    )
    group.add_argument('--overall_gpu_mem_ratio',
                       type=float,
                       default=0.8,
                       help='Used GPU memory in manager / total gpu memory.')
    group.add_argument('--overall_cpu_mem_ratio',
                       type=float,
                       default=0.8,
                       help='Used CPU memory in manager / total gpu memory.')
    group.add_argument('--warmup_gpu_chunk_mem_ratio',
                       type=float,
                       default=0.4,
                       help='warmup used gpu memory ratio.')
    group.add_argument('--margin_use_ratio',
                       type=float,
                       default=0.7,
                       help='GPu margin use ratio')
    group.add_argument(
        '--always_warmup',
        action='store_true',
        help='always warmup cancel dynamic GPU chunkable memory.')
    group.add_argument(
        '--use_gpu_fp32_convert_for_adam',
        action='store_true',
        help='use gpu fp32 convert for adam grad fp16 -> grad fp32.')
    return parser


def _add_test_bert_args(parser):
    group = parser.add_argument_group(title='test_bert')
    group.add_argument('--local_rank',
                       type=int,
                       default=None,
                       help='local rank passed from distributed launcher.')
    group.add_argument('--use_ckp',
                       dest='use_ckp',
                       action='store_true',
                       help='using checkpointing for memory saveing.')
    group.add_argument('--res_check',
                       dest='res_check',
                       action='store_true',
                       help='check results correctness of checkpointing.')
    group.add_argument('--use_fp16',
                       dest='use_fp16',
                       action='store_true',
                       help='using FP16 for training.')
    group.add_argument('--dist_plan',
                       type=str,
                       default='torch',
                       help='Distributed Plan [torch, ps, ds]')
    group.add_argument('--use_ds',
                       dest='use_ds',
                       action='store_true',
                       help='using DeepSpeed for training.')
    group.add_argument('--model_name',
                       type=str,
                       default='GPTsmall',
                       help='The model name.')
    return parser


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print('-------------------- arguments --------------------',
              flush=True)
        str_list = []
        for arg in vars(args):
            dots = '.' * (32 - len(arg))
            str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print('---------------- end of arguments ----------------', flush=True)


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description='PatrickStar Arguments')
    parser = _add_patrick_star_args(parser)
    parser = _add_test_bert_args(parser)
    args = parser.parse_args()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv("WORLD_SIZE", '1'))
    _print_args(args)
    return args


def test_bert_model_helper(args,
                           is_ckp: bool = False,
                           is_fp16: bool = False,
                           dist_plan: str = "torch",
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
    # 用单卡模拟多卡
    if args.use_fake_dist:
        rank = 0
    else:
        rank = args.local_rank
    # Avoid gpu0 use more memory.
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    device = torch.device(f'cuda:{rank}')

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

    # torch version
    if dist_plan == "ds":
        # TODO 测试并不正确
        import deepspeed
        model = BertForSequenceClassification(cfg)
        model.cuda(rank)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     betas=betas,
                                     eps=eps,
                                     weight_decay=weight_decay)
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=None,
            mpu=None,
            dist_init_required=True)
    elif dist_plan == "torch":
        model = BertForSequenceClassification(cfg)
        model.cuda(rank)
        if is_fp16:
            model = FP16_Module(model)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     betas=betas,
                                     eps=eps,
                                     weight_decay=weight_decay)
        if is_fp16:
            optimizer = FP16_Optimizer(optimizer)

        # DDP 不能要求模型部分在cpu部分在gpu
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[rank])
    elif dist_plan == "ps":
        assert is_fp16, f"use_ps must use fp16"

        def model_func():
            return BertForSequenceClassification(cfg)

        config = {
            # The same format as optimizer config of DeepSpeed
            # https://www.deepspeed.ai/docs/config-json/#optimizer-parameters
            "optimizer": {
                "type": "Adam",
                "params": {
                    "lr": lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "use_hybrid_adam": args.use_hybrid_adam
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 2**10,
                "initial_scale_power": 32,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "default_chunk_size": args.default_chunk_size,
            "use_fake_dist": args.use_fake_dist,
            "use_cpu_embedding": args.use_cpu_embedding
        }

        model, optimizer = initialize_engine(model_func=model_func,
                                             local_rank=rank,
                                             config=config)
    else:
        raise RuntimeError

    model_numel = get_ps_model_size(model)
    total_macs = estimate_bert_MAC(cfg, batch_size, sequence_length,
                                   model_numel)

    see_memory_usage(
        f"ckp {is_ckp} fp16 {is_fp16} dist_plan {dist_plan} after model init",
        force=True)

    data_loader = get_bert_data_loader(
        batch_size=batch_size,
        total_samples=10000,
        sequence_length=sequence_length,
        device=device,
        data_type=torch.half if is_fp16 else torch.float,
        is_distrbuted=True)

    loss_res = []

    start_time = time.time()

    logging.info(
        f"MAC {total_macs/1e9} GFlop, model numel {model_numel/1e9} B")

    for n, batch in enumerate(data_loader):
        step_start_time = time.time()

        output = model(input_ids=batch[0], labels=batch[1])
        loss = output.loss
        logits = output.logits
        # if torch.distributed.get_rank() == 0:
        logging.info(f"LOSS of step {n}: {loss.item()}")
        loss_res.append(loss.item())

        # logging.info(f'FWD finished moment {timer.moment()}')
        if dist_plan == "ds":
            model.backward(loss)
        elif dist_plan == "torch":
            if is_fp16:
                optimizer.zero_grad(set_grads_to_None=True)
                optimizer.backward(loss, update_master_grads=False)
                optimizer.update_master_grads()
            else:
                optimizer.zero_grad()
                loss.backward()
        elif dist_plan == "ps":
            if is_fp16:
                model.backward(loss)
            else:
                optimizer.zero_grad()
                loss.backward()

        # logging.info(f'BWD finished moment {timer.moment()}')
        if dist_plan == "ds":
            model.step()
        elif dist_plan == "torch" or dist_plan == "ps":
            optimizer.step()

        see_memory_usage(
            f"ckp {is_ckp} fp16 {is_fp16} dist_plan {dist_plan} after step {n}",
            force=True)

        step_elapse = time.time() - step_start_time
        if n == 0:
            logging.info(
                f"warmup ckp {is_ckp} fp16 {is_fp16} dist_plan {dist_plan}: step elapse {step_elapse} sec/iter, {total_macs/1e12/step_elapse} GFlops"
            )
        else:
            logging.info(
                f"ckp {is_ckp} fp16 {is_fp16} dist_plan {dist_plan}: step elapse {step_elapse} sec/iter, {total_macs/1e12/step_elapse} Tflops"
            )
        logging.info(f'model {model_numel/1e9}')

        if dist_plan == "ps":
            global_timer.my_timer.print()
            global_timer.data_move_cnter.print()

            global_timer.my_timer.reset()
            global_timer.data_move_cnter.reset()
        if n == stop_step: break

    elapse = time.time() - start_time

    # if is_ps:
    #     mgr = PatrickStarManager()
    #     mgr.show_mem_curve()

    logging.info("*" * 20)
    return loss_res


if __name__ == "__main__":
    os.environ["NCCL_DEBUG"] = "INFO"
    args = parse_args()
    use_ckp = args.use_ckp
    use_fp16 = args.use_fp16
    dist_plan = args.dist_plan
    # 检查结果正确性
    res_check = args.res_check
    # hidden_dim 1024, batch 16, seqence_leng 1024, ckp True.
    # PS is able to run the training, while PyTorch failed.

    logger.setLevel(logging.ERROR)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend='gloo' if args.use_fake_dist else 'nccl')

    world_size = torch.distributed.get_world_size()

    plan = args.model_name
    if res_check:
        plan = "Bert"
    if plan == "Bert":
        # 0.11B
        hidden_dim = 768
        sequence_length = 512
        num_layer = 6
        num_head = 12
    elif plan == 'Bertlarge':
        # 0.35B
        # PatrickStar and Torch都可以
        hidden_dim = 1024
        sequence_length = 512
        num_layer = 24
        num_head = 16
    elif plan == 'GPT2small':
        # 0.7B
        hidden_dim = 1536
        sequence_length = 128
        num_layer = 24
        num_head = 16
    elif plan == 'GPT2_1B':
        # 0.9B
        hidden_dim = 2048
        sequence_length = 1024
        num_layer = 20
        num_head = 16
    elif plan == 'megatron_1.3B':
        hidden_dim = 2048
        sequence_length = 1024
        num_layer = 24
        num_head = 32
    elif plan == 'GPT2_2B':
        # zero-offload
        hidden_dim = 2048
        sequence_length = 1024
        num_layer = 40
        num_head = 16
    elif plan == 'megatron_3.9B':
        # Table 4 in Megatron Paper
        hidden_dim = 2560
        sequence_length = 1024
        num_layer = 24
        num_head = 40
    elif plan == 'GPT2_4B':
        hidden_dim = 2304  #2048
        sequence_length = 1024
        num_layer = 64
        num_head = 16
    elif plan == 'GPT3_6B':
        # 6.7B model
        hidden_dim = 3072
        sequence_length = 1024
        num_layer = 53
        num_head = 16
    elif plan == 'GPT3_8B':
        # 6.7B model
        hidden_dim = 3072
        sequence_length = 1024
        num_layer = 72
        num_head = 16
    elif plan == 'GPT3_10B':
        hidden_dim = 4096
        sequence_length = 1024
        num_layer = 50
        num_head = 16
    elif plan == 'GPT3_11B':
        hidden_dim = 4096
        sequence_length = 1024
        num_layer = 55
        num_head = 16
    elif plan == 'GPT3_12B':
        hidden_dim = 4096
        sequence_length = 1024
        num_layer = 60
        num_head = 16
    elif plan == 'GPT3_13B':
        hidden_dim = 4096
        sequence_length = 1024
        num_layer = 65
        num_head = 16
    elif plan == 'GPT3_15B':
        hidden_dim = 4096
        sequence_length = 1024
        num_layer = 78
        num_head = 16
    else:
        raise RuntimeError(f"The model name {plan} is not valid!")
    if res_check:
        batch_size = 2
    else:
        batch_size = args.batch_size

    assert hidden_dim % num_head == 0
    logging.info(f'Benchmarking {plan}')
    if not res_check:
        # 训练参数，可以自己定义
        torch.manual_seed(0)
        loss_list = test_bert_model_helper(args=args,
                                           is_ckp=use_ckp,
                                           is_fp16=use_fp16,
                                           dist_plan=dist_plan,
                                           batch_size=batch_size,
                                           hidden_dim=hidden_dim,
                                           sequence_length=sequence_length,
                                           num_layer=num_layer,
                                           num_head=num_head,
                                           stop_step=5)
        print(loss_list)

    if res_check:
        torch.manual_seed(0)
        torch_res_list = test_bert_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=use_fp16,
            dist_plan="torch",
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_layer=num_layer,
            num_head=num_head)

        torch.cuda.empty_cache()
        print("*" * 50)
        torch.manual_seed(0)
        ps_res_list = test_bert_model_helper(args=args,
                                             is_ckp=use_ckp,
                                             is_fp16=use_fp16,
                                             dist_plan="ps",
                                             hidden_dim=hidden_dim,
                                             batch_size=batch_size,
                                             sequence_length=sequence_length,
                                             num_layer=num_layer,
                                             num_head=num_head)

        print('torch', torch_res_list)
        print('ps', ps_res_list)
        import numpy as np
        print(np.array(ps_res_list) - np.array(torch_res_list))
        for loss, loss_ref in zip(torch_res_list, ps_res_list):
            assert abs(loss - loss_ref) < 1e-4
