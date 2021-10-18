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

import argparse
import logging
import os
import time
from packaging import version

import torch
import numpy as np
import transformers
from transformers import BertConfig

# from transformers BertForSequenceClassification
# from ps_modeling_bert import BertForSequenceClassification


import patrickstar.utils.global_timer as global_timer
from data_loader import get_bert_data_loader
from patrickstar.profiler import profiler, torch_profiler
from patrickstar.runtime import initialize_engine
from patrickstar.utils import see_memory_usage
from patrickstar.utils.logging import logger
from patrickstar.utils.model_size_calculator import get_ps_model_size, estimate_bert_mac


def _add_patrick_star_args(parser):
    group = parser.add_argument_group(title="patrickstar")
    group.add_argument(
        "--use_fake_dist",
        dest="use_fake_dist",
        action="store_true",
        help="Using one GPU to stimulate multiple card.",
    )
    group.add_argument(
        "--default_chunk_size",
        type=int,
        default=32 * 1024 * 1024,
        help="Default Chunk Size in elements.",
    )
    group.add_argument(
        "--use_cpu_embedding",
        dest="use_cpu_embedding",
        action="store_true",
        help="Using CPU to perform Embedding and do not assign "
        "embedding params to chunks",
    )
    group.add_argument(
        "--release_after_init",
        action="store_true",
        help="Release the remote chunk after the whole initialization."
        "This would use more CPU memory during initialization, "
        "but may fix some errors relate to checkpoint loading or"
        "weight intialization.",
    )
    group.add_argument(
        "--use_hybrid_adam",
        action="store_true",
        help="Use hybrid adam optimization. "
        "By default ADAM is on CPU and run ADAM on GPU if possible.",
    )
    # Some hyperparams to tune when you failed to run a model.
    group.add_argument(
        "--always_warmup",
        action="store_true",
        help="Always warmup cancel dynamic GPU chunkable memory.",
    )
    group.add_argument(
        "--overall_gpu_mem_ratio",
        type=float,
        default=0.8,
        help="Used GPU memory in manager / total gpu memory.",
    )
    group.add_argument(
        "--overall_cpu_mem_ratio",
        type=float,
        default=0.8,
        help="Used CPU memory in manager / total gpu memory.",
    )
    group.add_argument(
        "--warmup_gpu_chunk_mem_ratio",
        type=float,
        default=0.4,
        help="warmup used gpu memory ratio.",
    )
    group.add_argument(
        "--margin_use_ratio", type=float, default=0.7, help="GPU margin use ratio"
    )
    group.add_argument(
        "--init_loss_scale", type=float, default=2 ** 6, help="initial loss scale"
    )
    return parser


def _add_test_bert_args(parser):
    group = parser.add_argument_group(title="test_bert")
    group.add_argument(
        "--batch_size", type=int, default=32, help="Batch size of input."
    )
    group.add_argument(
        "--local_rank",
        type=int,
        default=None,
        help="local rank passed from distributed launcher.",
    )
    group.add_argument(
        "--use_ckp",
        dest="use_ckp",
        action="store_true",
        help="using gradient checkpointing for memory saveing.",
    )
    group.add_argument(
        "--res_check",
        dest="res_check",
        action="store_true",
        help="check results correctness of checkpointing.",
    )
    group.add_argument(
        "--use_fp16",
        dest="use_fp16",
        action="store_true",
        help="using FP16 for training.",
    )
    group.add_argument(
        "--dist_plan",
        type=str,
        default="torch",
        help="Distributed Plan [torch, patrickstar]",
    )
    group.add_argument(
        "--model_name", type=str, default="GPTsmall", help="The model name."
    )
    group.add_argument(
        "--with_activation_offload",
        dest="with_activation_offload",
        action="store_true",
        help="Use activation offloading.",
    )
    return parser


def _add_lightseq_args(parser):
    group = parser.add_argument_group(title="test_light_seq")
    group.add_argument("--with_lightseq", action="store_true", help="use lightseq")
    return parser


def _print_args(args):
    """Print arguments."""
    if args.rank == 0:
        print("------------------- arguments -------------------", flush=True)
        str_list = []
        for arg in vars(args):
            dots = "." * (32 - len(arg))
            str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print("---------------- end of arguments ----------------", flush=True)


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="PatrickStar Arguments")
    parser = _add_patrick_star_args(parser)
    parser = _add_test_bert_args(parser)
    parser = _add_lightseq_args(parser)
    args = parser.parse_args()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    _print_args(args)
    return args


def print_model_config(hidden_dim, sequence_len, num_layer, num_head):
    if args.rank == 0:
        config_dict = {
            "hidden_dim": hidden_dim,
            "sequence_len": sequence_len,
            "num_layer": num_layer,
            "num_head": num_head,
        }
        print("------------------ model config ------------------", flush=True)
        str_list = []
        for key, value in config_dict.items():
            dots = "." * (32 - len(key))
            str_list.append("  {} {} {}".format(key, dots, value))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print("-------------- end of model config --------------", flush=True)


def test_bert_model_helper(
    args,
    is_ckp: bool = False,
    is_fp16: bool = False,
    dist_plan: str = "torch",
    batch_size=32,
    hidden_dim=768,
    sequence_length=256,
    num_layer=12,
    num_head=12,
    num_steps=5,
):
    logger.info(
        f'test a bert {"fp16" if is_fp16 else "fp32"} model '
        f'{"with checkpoint" if is_ckp else ""}'
    )

    # Use single card to imitate multicard.
    if args.use_fake_dist:
        rank = 0
    else:
        rank = args.local_rank

    if not args.with_activation_offload:
        from transformers import BertForSequenceClassification
    else:
        from ps_modeling_bert import BertForSequenceClassification

    # Avoid gpu0 use more memory.
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    device = torch.device(f"cuda:{rank}")

    bert_config = BertConfig(
        gradient_checkpointing=is_ckp,
        hidden_size=hidden_dim,
        intermediate_size=hidden_dim * 4,
        num_attention_heads=num_head,
        max_position_embeddings=sequence_length,
        num_hidden_layers=num_layer,
    )

    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-6
    weight_decay = 0

    profiler.start()
    if dist_plan == "patrickstar":
        if not is_fp16:
            logger.warning("PatrickStar will always use mixed precision training.")

        def model_func():
            model = BertForSequenceClassification(bert_config)
            if is_ckp and version.parse(transformers.__version__) >= version.parse(
                "4.11.0"
            ):
                model.gradient_checkpointing_enable()
            return model

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
                    "use_hybrid_adam": args.use_hybrid_adam,
                },
            },
            "fp16": {
                "enabled": True,
                "loss_scale": args.init_loss_scale,
                "initial_scale_power": 32,
                "loss_scale_window": 1000,
                "hysteresis": 2,
                "min_loss_scale": 1,
            },
            "default_chunk_size": args.default_chunk_size,
            "release_after_init": args.release_after_init,
            "use_fake_dist": args.use_fake_dist,
            "use_cpu_embedding": args.use_cpu_embedding,
        }

        model, optimizer = initialize_engine(
            model_func=model_func, local_rank=rank, config=config
        )
    else:
        from patrickstar.core.torch_profiler_hook import register_torch_profiler_hook

        model = BertForSequenceClassification(bert_config)
        register_torch_profiler_hook(model)
        if is_ckp and version.parse(transformers.__version__) >= version.parse(
            "4.11.0"
        ):
            model.gradient_checkpointing_enable()
        model.cuda(rank)
        model.train()
        if args.with_lightseq:
            from ls_hf_transformer_encoder_layer import inject_ls_enc_layer

            inject_ls_enc_layer(model, args, bert_config)
            print("Using Lightseq Kernels, all submodules includes:")

            def visit_and_register_hooks(module):
                is_child_node = True
                for _, submodule in module.named_children():
                    visit_and_register_hooks(submodule)
                    is_child_node = False
                if is_child_node:
                    print(f"module name {module.__class__.__name__}")

            visit_and_register_hooks(model)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        if is_fp16:
            scaler = torch.cuda.amp.GradScaler(
                init_scale=args.init_loss_scale,
                growth_factor=2,
                backoff_factor=0.5,
                growth_interval=1000,
            )

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    model_numel, model_num_param = get_ps_model_size(model)
    logger.info(f"Model size {model_numel / 1e9} B, total params: {model_num_param}")
    total_macs, nvidia_total_macs = estimate_bert_mac(
        bert_config, batch_size, sequence_length, model_numel
    )
    logger.info(f"Total MACs: {total_macs} TFlops")
    logger.info(f"NVIDIA total MACs: {nvidia_total_macs}")
    logger.debug(f"Diff csig/nvidia {total_macs / nvidia_total_macs}")

    see_memory_usage(
        f"After model init. using {dist_plan}, gradient checkpoint: {is_ckp}, fp16 {is_fp16}",
        force=True,
    )

    data_loader = get_bert_data_loader(
        batch_size=batch_size,
        total_samples=10000,
        sequence_length=sequence_length,
        device=device,
        data_type=torch.half if is_fp16 else torch.float,
        is_distrbuted=True,
    )

    loss_res = []

    print(f"MAC {total_macs / 1e9} GFlop, model param size: {model_numel / 1e9} B")

    for n, batch in enumerate(data_loader):
        if n == num_steps:
            break
        logger.info(f"Start Step {n} with {dist_plan}...")

        step_start_time = time.time()

        optimizer.zero_grad()

        if dist_plan == "patrickstar":
            output = model(input_ids=batch[0], labels=batch[1])
            loss = output[0]
            model.backward(loss)
            optimizer.step()
        else:
            if n == 1:
                torch_profiler.start()
            if n >= 1:
                torch_profiler.step_start.append(time.time())
            if is_fp16:
                with torch.cuda.amp.autocast():
                    output = model(input_ids=batch[0], labels=batch[1])
                loss = output[0]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(input_ids=batch[0], labels=batch[1])
                loss = output[0]
                loss.backward()
                optimizer.step()
            torch_profiler.step_end.append(time.time())

        logger.info(f"LOSS of step {n}: {loss.item()}")
        loss_res.append(loss.item())

        step_elapse = time.time() - step_start_time

        if args.rank == 0:
            see_memory_usage(
                f"After step {n}. using {dist_plan}, gradient checkpoint: {is_ckp}, fp16 {is_fp16}",
                force=True,
            )
            if dist_plan == "patrickstar":
                print(
                    f'{"[WARM UP] " if n == 0 else ""}'
                    f"Step elaspe {step_elapse} s, {total_macs / 1e12 / step_elapse} Tflops"
                )
                global_timer.my_timer.print()
                global_timer.data_move_cnter.print()

                global_timer.my_timer.reset()
                global_timer.data_move_cnter.reset()
            else:
                print(
                    f"Step elaspe {step_elapse} s, {total_macs / 1e12 / step_elapse} Tflops"
                )

        logger.info(f"End Step {n} with {dist_plan}.\n")

    profiler.end()
    if rank == 0:
        profiler.save("profile.pkl")
    torch_profiler.end()
    if rank == 0 and dist_plan == "torch":
        # Compare activation stats with:
        #     env GPU_NUM=1 DIST_PLAN="torch" ACT_OFFLOAD=0 CKP=1 BS=32 bash run_bert.sh
        #     env GPU_NUM=1 DIST_PLAN="torch" ACT_OFFLOAD=1 CKP=0 BS=32 bash run_bert.sh
        #     env GPU_NUM=1 DIST_PLAN="torch" ACT_OFFLOAD=0 CKP=0 BS=32 bash run_bert.sh
        torch_profiler.save("torch_profiler.pkl")
    logging.info("*" * 20)
    return loss_res


if __name__ == "__main__":
    # os.environ["NCCL_DEBUG"] = "INFO"
    args = parse_args()
    use_ckp = args.use_ckp
    use_fp16 = args.use_fp16
    dist_plan = args.dist_plan
    res_check = args.res_check

    # HIDDEN_DIM 1024, batch 16, seqence_len 1024, ckp True.
    # PatrickStar is able to run the training, while PyTorch failed.

    logger.setLevel(logging.WARNING)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo" if args.use_fake_dist else "nccl"
        )

    world_size = torch.distributed.get_world_size()

    MODEL_NAME = args.model_name
    if res_check:
        MODEL_NAME = "Bert"
    if MODEL_NAME == "Bert":
        # 0.11B
        HIDDEN_DIM = 768
        SEQ_LEN = 512
        NUM_LAYER = 6
        NUM_HEAD = 12
    elif MODEL_NAME == "Bertlarge":
        # 0.35B
        HIDDEN_DIM = 1024
        SEQ_LEN = 512
        NUM_LAYER = 24
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT2small":
        # 0.7B
        HIDDEN_DIM = 1536
        SEQ_LEN = 128
        NUM_LAYER = 24
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT2_1B":
        # 0.9B
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 20
        NUM_HEAD = 16
    elif MODEL_NAME == "megatron_1.3B":
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 24
        NUM_HEAD = 32
    elif MODEL_NAME == "GPT2_2B":
        # zero-offload
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 40
        NUM_HEAD = 16
    elif MODEL_NAME == "megatron_3.9B":
        # Table 4 in Megatron Paper
        HIDDEN_DIM = 2560
        SEQ_LEN = 1024
        NUM_LAYER = 24
        NUM_HEAD = 40
    elif MODEL_NAME == "GPT2_4B":
        HIDDEN_DIM = 2304  # 2048
        SEQ_LEN = 1024
        NUM_LAYER = 64
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT3_6B":
        # 6.7B model
        HIDDEN_DIM = 3072
        SEQ_LEN = 1024
        NUM_LAYER = 53
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT3_8B":
        # 6.7B model
        HIDDEN_DIM = 3072
        SEQ_LEN = 1024
        NUM_LAYER = 72
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT3_10B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 50
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT3_11B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 55
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT3_12B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 60
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT3_13B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 65
        NUM_HEAD = 16
    elif MODEL_NAME == "GPT3_15B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 78
        NUM_HEAD = 16
    else:
        raise RuntimeError(f"The model name {MODEL_NAME} is not valid!")
    if res_check:
        BATCH_SIZE = 2
    else:
        BATCH_SIZE = args.batch_size

    assert HIDDEN_DIM % NUM_HEAD == 0
    logging.info(f"Benchmarking {MODEL_NAME}")

    print_model_config(
        hidden_dim=HIDDEN_DIM,
        sequence_len=SEQ_LEN,
        num_layer=NUM_LAYER,
        num_head=NUM_HEAD,
    )

    if not res_check:
        torch.manual_seed(0)
        loss_list = test_bert_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=use_fp16,
            dist_plan=dist_plan,
            batch_size=BATCH_SIZE,
            hidden_dim=HIDDEN_DIM,
            sequence_length=SEQ_LEN,
            num_layer=NUM_LAYER,
            num_head=NUM_HEAD,
            num_steps=5,
        )
        print("*" * 20 + " LOSS " + "*" * 20)
        print(f"{loss_list}")

    if res_check:
        logging.warning(
            "Running to check result. This will use Bert model and batch size is 2."
        )

        NUM_STEPS = 5

        torch.manual_seed(0)
        torch_res_list = test_bert_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=False,
            dist_plan="torch",
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            sequence_length=SEQ_LEN,
            num_layer=NUM_LAYER,
            num_head=NUM_HEAD,
            num_steps=NUM_STEPS,
        )

        torch.cuda.empty_cache()
        logging.info("-" * 50)

        torch.manual_seed(0)
        autocast_res_list = test_bert_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=True,
            dist_plan="torch",
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            sequence_length=SEQ_LEN,
            num_layer=NUM_LAYER,
            num_head=NUM_HEAD,
            num_steps=NUM_STEPS,
        )

        torch.cuda.empty_cache()
        logging.info("-" * 50)

        torch.manual_seed(0)
        ps_res_list = test_bert_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=use_fp16,
            dist_plan="patrickstar",
            hidden_dim=HIDDEN_DIM,
            batch_size=BATCH_SIZE,
            sequence_length=SEQ_LEN,
            num_layer=NUM_LAYER,
            num_head=NUM_HEAD,
            num_steps=NUM_STEPS,
        )

        print("-" * 20 + " LOSS " + "-" * 20)
        print(f"torch fp32 : {torch_res_list}")
        print(f"autocast   : {autocast_res_list}")
        print(f"patrickstar: {ps_res_list}")

        def diff(array):
            return list(np.array(ps_res_list) - np.array(array))

        print("-" * 20 + " DIFF " + "-" * 20)
        print(f"vs torch fp32: {diff(torch_res_list)}")
        print(f"vs autocast  : {diff(autocast_res_list)}")
