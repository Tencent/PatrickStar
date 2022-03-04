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

from patrickstar.utils.logging import logger, log_dist
from model_builder import build_transformer_model
from ps_config import get_patrickstar_config
from parse_args import parse_args
from patrickstar.core import PatrickStarClient
from patrickstar.core import PSPreProcessCtx
import time
from patrickstar.utils.distributed import get_rank

MB_NUM = 1024 * 1024
GB_NUM = 1024 * MB_NUM

HARDWARE_SETTING_JSON = {
    "per_cpu_mem": 240 * GB_NUM,
    "per_gpu_mem": 32 * GB_NUM,
    "global_gpu_num": 1,
    "gloabl_cpu_num": 1,
    "local_gpu_num": 1,
    "local_cpu_num": 1,
}


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
    per_gpu_mem = HARDWARE_SETTING_JSON.get("per_gpu_mem")
    per_cpu_mem = HARDWARE_SETTING_JSON.get("per_cpu_mem")
    global_gpu_num = HARDWARE_SETTING_JSON.get("global_gpu_num")
    global_cpu_num = HARDWARE_SETTING_JSON.get("gloabl_cpu_num")
    ava_per_gpu_mem = (
        per_gpu_mem
        * config.get("overall_gpu_mem_ratio", 0.8)
        * config.get("warmup_gpu_chunk_mem_ratio", 0.1)
    )

    ava_per_cpu_mem = per_cpu_mem * config.get("overall_cpu_mem_ratio", 0.8)

    # GPU mem has to host at least two chunks.
    if ava_per_gpu_mem < chunk_size * 2:
        logger.error(
            "chunk is unable to be fitted in GPU during warmup!\n"
            "GPU Mem %.2f MB vs. Two Chunks %.2f MB",
            ava_per_gpu_mem / MB_NUM,
            chunk_size * 2 / MB_NUM,
        )
        return False

    # CPU + GPU shall not exceed the 14M (M numel of param)
    overall_cpu_gpu_mem = (
        ava_per_gpu_mem * global_gpu_num + ava_per_cpu_mem * global_cpu_num
    )
    need_mem = overall_chunk_size / 6 * 14
    if overall_cpu_gpu_mem < need_mem:
        logger.error(
            "Overall chunks can't fit in memory of CPU+GPU " "%.2f MB vs. %.2f MB",
            overall_cpu_gpu_mem / MB_NUM,
            need_mem / MB_NUM,
        )
        return False

    logger.info(
        "Evaluated chunk size %d Melem"
        "ava_per_gpu_mem %.2f MB, "
        "ava_per_cpu_mem %.2f MB, "
        "need_mem %.2f MB\n",
        args.chunk_size / MB_NUM,
        ava_per_gpu_mem / MB_NUM,
        ava_per_cpu_mem / MB_NUM,
        need_mem / MB_NUM,
    )
    return True


def get_param_used_chunk_size(args, config, model_func):
    """
    return overall chunk size of param fp16 and param fp32.
    as well as the memory utilization of chunks.
    """
    client = PatrickStarClient(
        local_rank=args.local_rank,
        chunk_size=args.chunk_size,
        config=config.get("client", None),
    )
    start_time = time.time()
    try:
        with PSPreProcessCtx(
            client=client,
            release_after_init=args.release_after_init,
            not_init=True,
        ):
            model = model_func()
    except Exception:
        logger.error("PSPreProcessCtx failed")
        return -1, -1
    end_time = time.time()
    log_dist(f"PSPreProcessCtx Model Constructing elapse {end_time - start_time}")
    del model

    overall_chunk_size, util = client.get_overall_chunk_size()
    if chunk_schema_valid_check(
        args,
        config["client"]["mem_tracer"],
        args.chunk_size,
        overall_chunk_size,
    ):

        return overall_chunk_size, util
    else:
        logger.error("Chunk schema validation check failed!")
        return overall_chunk_size, -1


def evaluate_chunk_size(args):
    """
    Evaluate the current training task defined by the args.
    write the chunk memory usage to the file.
    """
    # Avoid gpu0 use more memory.
    # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113
    torch.cuda.set_device(args.local_rank)
    torch.cuda.empty_cache()

    lr = 0.001
    betas = (0.9, 0.999)
    eps = 1e-6
    weight_decay = 0

    model_func, sequence_length = build_transformer_model(args)
    config = get_patrickstar_config(
        args, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay
    )

    overall_chunk_size, utils = get_param_used_chunk_size(args, config, model_func)

    logger.info(
        "chunk uses %.2f MB, utilization %.2f \n", overall_chunk_size / MB_NUM, utils
    )
    logger.info(f"writing to {args.slog_file}\n")

    if get_rank() == 0:
        with open(f"{args.slog_file}", "a+") as fh:
            fh.write(
                f"{args.chunk_size/1024/1024} {overall_chunk_size/1024/1024}, {utils}\n"
            )


if __name__ == "__main__":
    args = parse_args()
    logger.setLevel(logging.INFO)
    torch.manual_seed(0)
    evaluate_chunk_size(args=args)
