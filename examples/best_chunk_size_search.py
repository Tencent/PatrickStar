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
import torch
from patrickstar.utils import see_memory_usage
from patrickstar.utils.logging import logger
from model_builder import build_transformer_model
from ps_config import get_patrickstar_config
from parse_args import parse_args
from patrickstar.core import PatrickStarClient
from patrickstar.core import PSPreProcessCtx

from patrickstar.utils.distributed import get_local_world_size, get_rank
from patrickstar.utils.memory import get_memory_info


def chunk_schema_valid_check(args, config, chunk_size, overall_chunk_size):
    """
    check validation of a chunk schema, given the overall chunk size
    args:
        @args: cmd args
        @config: client config
        @chunk_size: the chunk size in numel
        @overall_chunk_size: the overall chunk size used for param fp16
    returns:
        bool: is the chunk schema valid
    """
    mem_info = get_memory_info()
    local_world_size = get_local_world_size()
    overall_gpu_mem = torch.cuda.get_device_properties(
        args.local_rank
    ).total_memory * config.get("overall_gpu_mem_ratio", 0.8)
    overall_cpu_mem = (
        mem_info.total * config.get("overall_cpu_mem_ratio", 0.8) / local_world_size
    )
    warmup_used_gpu_mem = overall_gpu_mem * config.get(
        "warmup_gpu_chunk_mem_ratio", 0.1
    )

    logger.info(
        f"warmup_used_gpu_mem {warmup_used_gpu_mem}, "
        f"overall_cpu_mem {overall_cpu_mem}, "
        f"overall_chunk_size {overall_chunk_size}"
    )
    if warmup_used_gpu_mem < chunk_size * 2:
        logger.info("chunk is unable to be fitted in GPU during warmup")
        return False

    if warmup_used_gpu_mem + overall_cpu_mem < overall_chunk_size / 6 * 14:
        logger.info("overall chunks is not able to fit in CPU + GPU")
        return False
    return True


def get_param_used_chunk_size(args, config, model_func):
    """
    return overall chunk size of param fp16 and param fp32.
    as well as the memory utilization of chunks.
    """
    client = PatrickStarClient(
        rank=args.local_rank,
        default_chunk_size=args.default_chunk_size,
        config=config.get("client", None),
    )

    try:
        with PSPreProcessCtx(
            client=client,
            dtype=torch.float,
            release_after_init=args.release_after_init,
            use_cpu_embedding=args.use_cpu_embedding,
            not_init=True,
        ):
            model = model_func()
    except Exception:
        return -1, -1

    del model

    overall_chunk_size, util = client.get_overall_chunk_size()
    if chunk_schema_valid_check(
        args,
        config["client"]["mem_tracer"],
        args.default_chunk_size,
        overall_chunk_size,
    ):
        return overall_chunk_size, util
    else:
        return -1, -1


def evaluate_chunk_size(
    args,
    is_ckp: bool = False,
    is_fp16: bool = False,
    dist_plan: str = "torch",
    num_steps=5,
):
    """
    Evaluate the current training task defined by the args.
    write the chunk memory usage to the file.
    """
    logger.info(
        f'test a bert {"fp16" if is_fp16 else "fp32"} model '
        f'{"with checkpoint" if is_ckp else ""}'
    )

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

    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-6
    weight_decay = 0

    model_func, sequence_length = build_transformer_model(args)
    config = get_patrickstar_config(
        args, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )

    see_memory_usage(
        f"before get_param_used_chunk_size for {args.default_chunk_size/1024/1024} MB",
        True,
        "MB",
    )

    overall_chunk_size, utils = get_param_used_chunk_size(args, config, model_func)

    see_memory_usage(
        f"after get_param_used_chunk_size for {args.default_chunk_size/1024/1024} MB",
        True,
        "MB",
    )

    logger.info(f"{overall_chunk_size}, {utils}\n")
    logger.info(f"writing to {args.slog_file}\n")

    if get_rank() == 0:
        with open(f"{args.slog_file}", "a+") as fh:
            fh.write(
                f"{args.default_chunk_size/1024/1024} {overall_chunk_size/1024/1024}, {utils}\n"
            )


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

    torch.manual_seed(0)
    loss_list = evaluate_chunk_size(
        args=args,
        is_ckp=use_ckp,
        is_fp16=use_fp16,
        dist_plan=dist_plan,
        num_steps=5,
    )
