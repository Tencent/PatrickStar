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

import logging
import time

import torch
import numpy as np
from apex import amp

from patrickstar.runtime import initialize
from patrickstar.utils import see_memory_usage, get_world_size, global_timer
from patrickstar.utils.logging import log_dist, logger
from patrickstar.utils.model_size_calculator import get_ps_model_size
from model_builder import build_transformer_model
from parse_args import parse_args
from config import get_patrickstar_config


def get_bert_data_loader(
    batch_size,
    total_samples,
    sequence_length,
    device,
):
    train_data = torch.randint(
        low=0,
        high=1000,
        size=(total_samples, sequence_length),
        device=device,
        dtype=torch.long,
    )
    train_label = torch.randint(
        low=0, high=2, size=(total_samples,), device=device, dtype=torch.long
    )
    train_dataset = torch.utils.data.TensorDataset(train_data, train_label)
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler
    )
    return train_loader


def test_transformer_model_helper(
    args,
    dist_plan: str,
    num_steps,
):
    rank = args.local_rank

    # Avoid gpu0 use more memory.
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-6
    weight_decay = 0

    config = get_patrickstar_config(
        args, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )

    model_func, sequence_length = build_transformer_model(args)
    if dist_plan == "patrickstar":
        model, optimizer = initialize(
            model_func=model_func, local_rank=rank, config=config
        )
    else:
        model = model_func()

        model.cuda(rank)
        model.train()

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
        )
        model, optimizer = amp.initialize(
            model,
            optimizer,
            opt_level="O2",
            loss_scale="dynamic",
            max_loss_scale=config["fp16"]["init_scale"],
        )
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    model_numel, model_num_param = get_ps_model_size(model)
    log_dist(f"Model size {model_numel / 1e9} B, total params: {model_num_param}")
    total_macs = model_numel * args.batch_size * sequence_length * 2 * 4
    log_dist(f"Total MACs: {total_macs / 1024 ** 4} TFlops")

    see_memory_usage(f"After model init. using {dist_plan}")

    # load data, here we generate random data for benchmarking.
    data_loader = get_bert_data_loader(
        batch_size=args.batch_size,
        total_samples=10000,
        sequence_length=sequence_length,
        device=device,
    )

    loss_res = []

    print(f"model param size: {model_numel / 1024 ** 3} B")

    for n, batch in enumerate(data_loader):
        if n == num_steps:
            break
        if n == num_steps - 1:
            global_timer.start()

        # You may need to empty_cache for really large models.
        # torch.cuda.empty_cache()
        step_start = time.time()

        if dist_plan == "patrickstar":
            output = model(input_ids=batch[0], labels=batch[1])
            loss = output[0]
            model.backward(loss)
            model.step()
            model.zero_grad()
        else:
            output = model(input_ids=batch[0], labels=batch[1])
            loss = output["loss"]
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print(f"LOSS of step {n}: {loss.item()}")
        loss_res.append(loss.item())

        step_elapse = time.time() - step_start

        if args.rank == 0:
            world_size = get_world_size()
            if dist_plan == "patrickstar":
                print(
                    f'{"[WARM UP] " if n == 0 else ""}'
                    f"Step {n} elaspe {step_elapse} s, "
                    f"{total_macs / 1e12 / step_elapse} Tflops Per GPU "
                    f"{args.batch_size * world_size/step_elapse} SamplesPerSec"
                )
                if n == num_steps - 1:
                    global_timer.print()
                    global_timer.reset()
            else:
                print(
                    f"Step {n} elaspe {step_elapse} s, "
                    f"{total_macs / 1e12 / step_elapse} Tflops Per GPU "
                    f"{args.batch_size * world_size/step_elapse} SamplesPerSec"
                )
    return loss_res


if __name__ == "__main__":
    args = parse_args()
    res_check = args.res_check

    # You could set the logger level to INFO to view more log.
    logger.setLevel(logging.INFO)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    world_size = torch.distributed.get_world_size()

    if not res_check:
        torch.manual_seed(0)
        loss_list = test_transformer_model_helper(
            args=args,
            dist_plan="patrickstar",
            num_steps=5,
        )
        print("*" * 20 + " LOSS " + "*" * 20)
        print(f"{loss_list}")
    else:
        logging.warning(
            "Running to check result. This will use Bert model and batch size is 2."
        )

        args.model_name = "Bert"
        args.batch_size = 2
        NUM_STEPS = 10

        torch.manual_seed(0)
        torch_res_list = test_transformer_model_helper(
            args=args,
            dist_plan="apex",
            num_steps=NUM_STEPS,
        )

        torch.cuda.empty_cache()
        logging.info("-" * 50)

        torch.manual_seed(0)
        ps_res_list = test_transformer_model_helper(
            args=args,
            dist_plan="patrickstar",
            num_steps=NUM_STEPS,
        )

        print("-" * 20 + " LOSS " + "-" * 20)
        print(f"apex O2: {torch_res_list}")
        print(f"patrickstar: {ps_res_list}")

        def diff(a, b, dtype):
            np_a = np.array(a, dtype=dtype)
            np_b = np.array(b, dtype=dtype)
            return list(np_a - np_b)

        print("-" * 20 + " DIFF " + "-" * 20)
        print(f"vs torch: {diff(ps_res_list, torch_res_list, np.float16)}")
