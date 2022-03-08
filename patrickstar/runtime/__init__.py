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

import time

import torch

from patrickstar.core import PSPreProcessCtx, PatrickStarClient
from patrickstar.runtime.engine import PatrickStarEngine
from patrickstar.utils import logger, log_dist

DEFAULT_CHUNK_SIZE = 32 * 1024 * 1024


def initialize_engine(model_func, local_rank, config=None, client=None):
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
    if isinstance(model_func, torch.nn.Module):
        logger.debug(
            "Passing nn.Module into initialize_engine. "
            "Make sure you have intialized the model within PSPreProcessCtx"
        )
        assert client is not None, "Must pass the client when passing a nn.Module."
        model = model_func
    else:
        assert callable(model_func), "model_func need to be callable."

        if config is None:
            chunk_size = DEFAULT_CHUNK_SIZE
            release_after_init = False
        else:
            chunk_size = config.get("chunk_size", DEFAULT_CHUNK_SIZE)
            release_after_init = config.get("release_after_init", False)

        client = PatrickStarClient(
            local_rank=local_rank,
            chunk_size=chunk_size,
            config=config.get("client", None),
        )

        start_time = time.time()
        log_dist("Begin initializing model...")
        with PSPreProcessCtx(
            client=client,
            release_after_init=release_after_init,
        ):
            model = model_func()
        end_time = time.time()
        log_dist(f"Finish initializing model in {end_time  - start_time} s")

    engine = PatrickStarEngine(model=model, client=client, config=config)
    client.start_mem_tracer()
    return (engine, engine.optimizer)
