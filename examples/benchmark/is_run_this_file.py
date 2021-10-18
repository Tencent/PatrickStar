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
from process_logs import is_run_this_file


def add_args(parser):
    group = parser.add_argument_group(title="patrickstar")
    group.add_argument(
        "--file",
        type=str,
        help="file name.",
    )
    group.add_argument(
        "--path",
        type=str,
        help="path name.",
    )
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PatrickStar Arguments")
    parser = add_args(parser)
    args = parser.parse_args()
    IS_RUN = is_run_this_file(args.path, args.file, {}, {})

    if IS_RUN:
        print(1)
    else:
        print(0)
