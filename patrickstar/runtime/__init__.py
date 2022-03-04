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
            rank=local_rank,
            chunk_size=chunk_size,
            config=config.get("client", None),
        )

        start_time = time.time()
        log_dist("begin initialize the model parameters...")
        with PSPreProcessCtx(
            client=client,
            release_after_init=release_after_init,
        ):
            model = model_func()
        end_time = time.time()
        log_dist(
            f"finished initialized the model parameters... {end_time  - start_time} s"
        )

    engine = PatrickStarEngine(model=model, client=client, config=config)
    client.start_mem_tracer()
    return (engine, engine.optimizer)
