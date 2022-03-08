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

from patrickstar.core.parameter import is_registered


def get_ps_model_size(model):
    numel = 0
    param_cnt = 0
    for _, param in model.named_parameters(recurse=True):
        if is_registered(param):
            numel += param.ps_attr.numel
        else:
            numel += param.numel()
        param_cnt += 1
    return numel, param_cnt


def estimate_bert_mac(config, batch_size, sequence_length, model_size):
    nvidia_total_macs = (
        96
        * batch_size
        * sequence_length
        * config.num_hidden_layers
        * config.hidden_size ** 2
        * (
            1
            + sequence_length / (6 * config.hidden_size)
            + config.vocab_size / (16 * config.num_hidden_layers * config.hidden_size)
        )
    )

    tera_flops = model_size * batch_size * sequence_length * 2 * 4
    return tera_flops, nvidia_total_macs
