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
from .engine import PatrickStarEngine
from .init_context import Init
from ..manager import PatrickStarManager
from patrickstar.core import PSPreProcessCtx


def initialize_engine(model_func, client, config=None):
    """Initialize the PatrickStar Engine.
    Arguments:
        model_func: Required: nn.module class before apply any wrappers
        client: Required: PatrickStarClient for orchestrating chunks.
        config: Optional: config json for optimizer.
    Returns:
        A tuple of ``engine`` and ``optimizer``
        * ``engine``: PatrickStar runtime engine which wraps the client model for distributed training.
        * ``optimizer``: Wrapped optimizer if a user defined ``optimizer`` is supplied, or if
          optimizer is specified in json config else ``None``.
    """
    if not callable(model_func):
        raise ValueError("model_func need to be callable.")

    with PSPreProcessCtx(client=client, dtype=torch.float):
        model = model_func()

    engine = PatrickStarEngine(model=model, client=client, config=config)

    # 开启预热优化
    mgr = PatrickStarManager()
    mgr.start_train(
        is_warmup=True,
        param_fp16_chunk_size=client.get_param_fp16_chunks_mem_size(),
        chunk_size=client.default_chunk_size)

    return (engine, engine.optimizer)
