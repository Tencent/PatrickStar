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
from patrickstar.deepspeed_helper.global_vars import get_args
from patrickstar.core.parameter import is_param_registered


def get_ps_model_size(model):
    numel = 0
    param_cnt = 0
    for name, param in model.named_parameters(recurse=True):
        if is_param_registered(param):
            numel += param.ps_attr.numel
        else:
            numel += param.numel()
        param_cnt += 1
    # numel *= args.world_size
    print(f"PS model size {numel/1e9} B, param cnt {param_cnt}")
    return numel


def estimate_bert_MAC(config, batch_size, sequence_length, model_size):
    B = batch_size
    S = sequence_length
    V = config.vocab_size
    N = config.num_attention_heads
    H = config.hidden_size
    L = config.num_hidden_layers
    P = config.max_position_embeddings
    cisg_total_macs = 72 * B * S * N * H**2 + 12 * B * N * H * S**2
    nvidia_total_macs = 96 * B * S * L * H**2 * (1 + S / (6 * H) + V /
                                                 (16 * L * H))

    tera_flops = model_size * batch_size * sequence_length * 2 * 4
    print(f'tera_flops total MACs {tera_flops}')
    print(f'nvidia total MACs {nvidia_total_macs}')
    print(f'diff csig/nvidia {tera_flops / nvidia_total_macs}')
    return tera_flops
