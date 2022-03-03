# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import logging
import time

import torch
import numpy as np

import patrickstar.utils.global_timer as global_timer
from data_loader import get_bert_data_loader
from patrickstar.profiler import profiler
from patrickstar.runtime import initialize_engine
from patrickstar.utils import see_memory_usage, get_world_size
from patrickstar.utils.logging import log_dist, logger
from patrickstar.utils.model_size_calculator import get_ps_model_size
from model_builder import build_transformer_model
from parse_args import parse_args
from ps_config import get_patrickstar_config


def test_transformer_model_helper(
    args,
    is_ckp: bool,
    is_fp16: bool,
    dist_plan: str,
    num_steps,
):
    # Use single card to simulate multicard. Used when you are poor and
    # no more GPU avaiable.
    if args.use_fake_dist:
        rank = 0
    else:
        rank = args.local_rank

    # Avoid gpu0 use more memory.
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{rank}")

    if args.with_mem_profiler:
        print("start memory profiler")
        profiler.start()

    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-6
    weight_decay = 0

    model_func, sequence_length = build_transformer_model(args)
    if dist_plan == "patrickstar":
        if not is_fp16:
            logger.warning("PatrickStar will always use mixed precision training.")
        config = get_patrickstar_config(
            args, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        model, optimizer = initialize_engine(
            model_func=model_func, local_rank=rank, config=config
        )
    else:
        model = model_func()
        if args.with_mem_profiler:
            from patrickstar.core.torch_profiler_hook import (
                register_torch_profiler_hook,
            )

            register_torch_profiler_hook(model)

        model.cuda(rank)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )

        if is_fp16:
            scaler = torch.cuda.amp.GradScaler(
                init_scale=2 ** args.init_loss_scale_power,
                growth_factor=2,
                backoff_factor=0.5,
                growth_interval=1000,
            )

        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    model_numel, model_num_param = get_ps_model_size(model)
    log_dist(f"Model size {model_numel / 1e9} B, total params: {model_num_param}")
    total_macs = model_numel * args.batch_size * sequence_length * 2 * 4
    log_dist(f"Total MACs: {total_macs/1024/1024/1024/1024} TFlops")

    see_memory_usage(
        f"After model init. using {dist_plan}, gradient checkpoint: {is_ckp}, fp16 {is_fp16}",
        force=True,
    )

    # load data, here we generate random data for benchmarking.
    data_loader = get_bert_data_loader(
        batch_size=args.batch_size,
        total_samples=10000,
        sequence_length=sequence_length,
        device=device,
        data_type=torch.half if is_fp16 else torch.float,
        is_distrbuted=True,
    )

    loss_res = []

    print(f"model param size: {model_numel / 1e9} B")

    for n, batch in enumerate(data_loader):
        if n == num_steps:
            break
        # You may need to empty_cache for really large models.
        torch.cuda.empty_cache()
        log_dist(f"Start Step {n} with {dist_plan}...")

        step_start_time = time.time()
        # Only collect running time of the last iteration.
        if n == num_steps - 1:
            global_timer.my_timer.start()
        optimizer.zero_grad()
        if args.with_mem_profiler:
            if n == 1:
                profiler.warmup_finish()

        if dist_plan == "patrickstar":
            output = model(input_ids=batch[0], labels=batch[1])
            loss = output[0]
            model.backward(loss)
            optimizer.step()
        else:
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

        print(f"LOSS of step {n}: {loss.item()}")
        loss_res.append(loss.item())

        step_elapse = time.time() - step_start_time

        if args.rank == 0:
            see_memory_usage(
                f"After step {n}. using {dist_plan}, gradient checkpoint: {is_ckp}, fp16 {is_fp16}",
                force=True,
            )
            world_size = get_world_size()
            if dist_plan == "patrickstar":
                print(
                    f'{"[WARM UP] " if n == 0 else ""}'
                    f"Step {n} elaspe {step_elapse} s, "
                    f"{total_macs / 1e12 / step_elapse} Tflops Per GPU "
                    f"{args.batch_size * world_size/step_elapse} SamplesPerSec"
                )
                if n == num_steps - 1:
                    global_timer.my_timer.print()
                    global_timer.data_move_cnter.print()

                    global_timer.my_timer.reset()
                    global_timer.data_move_cnter.reset()
            else:
                print(
                    f"Step {n} elaspe {step_elapse} s, "
                    f"{total_macs / 1e12 / step_elapse} Tflops Per GPU "
                    f"{args.batch_size * world_size/step_elapse} SamplesPerSec"
                )

        log_dist(f"End Step {n} with {dist_plan}.\n")

    if args.with_mem_profiler:
        profiler.end()
        if rank == 0:
            profiler.save(
                f"{dist_plan}_{args.model_name}_bs_{args.batch_size}_"
                f"ckp_{is_ckp}_offload_{args.with_activation_offload}_profile.pkl"
            )
    return loss_res


if __name__ == "__main__":
    args = parse_args()
    use_ckp = args.use_ckp
    use_fp16 = args.use_fp16
    dist_plan = args.dist_plan
    res_check = args.res_check

    # You could set the logger level to INFO to view more runtime
    # information.
    logger.setLevel(logging.INFO)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo" if args.use_fake_dist else "nccl"
        )

    world_size = torch.distributed.get_world_size()

    if not res_check:
        torch.manual_seed(0)
        loss_list = test_transformer_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=use_fp16,
            dist_plan=dist_plan,
            num_steps=20,
        )
        print("*" * 20 + " LOSS " + "*" * 20)
        print(f"{loss_list}")

    if res_check:
        logging.warning(
            "Running to check result. This will use Bert model and batch size is 2."
        )

        args.model_name = "Bert"
        args.batch_size = 2
        NUM_STEPS = 10

        torch.manual_seed(0)
        torch_res_list = test_transformer_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=False,
            dist_plan="torch",
            num_steps=NUM_STEPS,
        )

        torch.cuda.empty_cache()
        logging.info("-" * 50)

        torch.manual_seed(0)
        autocast_res_list = test_transformer_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=True,
            dist_plan="torch",
            num_steps=NUM_STEPS,
        )

        torch.cuda.empty_cache()
        logging.info("-" * 50)

        torch.manual_seed(0)
        ps_res_list = test_transformer_model_helper(
            args=args,
            is_ckp=use_ckp,
            is_fp16=False,  # use_fp16,
            dist_plan="patrickstar",
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
