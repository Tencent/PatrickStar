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

import fire
import torch

from patrickstar.utils import logger


def merge_checkpoint(pattern, num):
    merged_state_dict = {}
    for i in range(num):
        filename = pattern.replace("*", f"{i}")
        merged_state_dict.update(torch.load(filename))

    for k, v in merged_state_dict.items():
        print(k, v)

    merged_filename = pattern.replace("*", "merged")
    logger.warning(f"Merged checkpoint will be saved to {merged_filename}")
    torch.save(merged_state_dict, merged_filename)


if __name__ == "__main__":
    fire.Fire(merge_checkpoint)
