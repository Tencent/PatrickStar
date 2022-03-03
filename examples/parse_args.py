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

import argparse
import os


def _add_patrick_star_args(parser):
    group = parser.add_argument_group(title="patrickstar")
    group.add_argument(
        "--chunk_size",
        type=int,
        default=32 * 1024 * 1024,
        help="Default Chunk Size in elements.",
    )
    group.add_argument(
        "--release_after_init",
        action="store_true",
        help="Release the remote chunk after the whole initialization."
        "This would use more CPU memory during initialization, "
        "but may fix some errors relate to checkpoint loading or"
        "weight intialization.",
    )
    # Some hyperparams to tune when you failed to run a model.
    group.add_argument(
        "--with_static_partition",
        action="store_true",
        help="Use static partition for model data on CPU and GPU.",
    )
    group.add_argument(
        "--with_mem_profiler",
        action="store_true",
        help="Profiling memory usage.",
    )
    group.add_argument(
        "--init_loss_scale_power",
        type=float,
        default=10,
        help="initial loss scale power",
    )
    group.add_argument(
        "--with_async_mem_monitor",
        action="store_true",
        help="Use async memory monitor.",
    )
    group.add_argument(
        "--slog_file",
        type=str,
        default="./slog_file/tmp.txt",
        help="The file to record chunk size serach log.",
    )
    return parser


def _add_general_opt_args(parser):
    group = parser.add_argument_group(title="test_bert")
    group.add_argument(
        "--use_ckp",
        dest="use_ckp",
        action="store_true",
        help="using gradient checkpointing for memory saveing.",
    )
    group.add_argument(
        "--with_activation_offload",
        dest="with_activation_offload",
        action="store_true",
        help="Use activation offloading.",
    )
    group.add_argument(
        "--with_tiling_linear",
        action="store_true",
        help="Use linear tiling.",
    )
    return parser


def _add_test_config_args(parser):
    group = parser.add_argument_group(title="test_config")
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
        "--model_type",
        type=str,
        default="BERT",
        help="The type of the backbone of the model.",
    )
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
    parser = _add_test_config_args(parser)
    parser = _add_general_opt_args(parser)
    args = parser.parse_args()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    _print_args(args)
    return args
