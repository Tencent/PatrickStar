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
"""
python ../launcher/runner.py --num_nodes 1 --num_gpus 1 dist_init_test.py
"""
import torch
from runtime import initialize_engine, init_context, Init
from tests.simple_net import SimpleModel

hidden_dim = 4

# 初始化模型参数
with Init():
    model = SimpleModel(hidden_dim=hidden_dim)

rank = torch.distributed.get_rank()
for param in model.named_parameters():
    print(f'rank {rank}', param)

# 初始化计算引擎，不常用
model, _, _, _ = initialize_engine(args=None,
                                   model=model,
                                   model_parameters=model.parameters())
