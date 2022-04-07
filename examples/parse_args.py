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
import os


def add_patrickstar_args(parser):
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
    return parser


def add_test_config_args(parser):
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
        "--model_name", type=str, default="GPTsmall", help="The model name."
    )
    group.add_argument(
        "--model_type",
        type=str,
        default="BERT",
        help="The type of the backbone of the model.",
    )
    return parser


def print_args(args):
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
    parser = add_patrickstar_args(parser)
    parser = add_test_config_args(parser)
    args = parser.parse_args()
    args.rank = int(os.getenv("RANK", "0"))
    args.world_size = int(os.getenv("WORLD_SIZE", "1"))
    print_args(args)
    return args
