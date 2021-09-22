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

import logging
import sys

import torch.distributed as dist

from .distributed import get_rank


class LoggerFactory:
    @staticmethod
    def create_logger(name=None, level=logging.WARNING):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger
        Raises:
            ValueError is name is None
        """

        if name is None:
            raise ValueError("name for logger cannot be None")

        # formatter = logging.Formatter(
        #     "[%(asctime)s] [%(levelname)s] "
        #     "[%(filename)s:%(lineno)d:%(funcName)s] %(message)s")

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger_.addHandler(ch)
        return logger_


logger = LoggerFactory.create_logger(name="PatrickStar", level=logging.WARNING)


def log_dist(message, ranks=None, level=logging.INFO):
    """Log message when one of following condition meets
    + not dist.is_initialized()
    + dist.get_rank() in ranks if ranks is not None or ranks = [-1]
    Args:
        message (str)
        ranks (list)
        level (int)
    """
    should_log = not dist.is_initialized()
    ranks = ranks or []
    my_rank = dist.get_rank() if dist.is_initialized() else -1
    if ranks and not should_log:
        should_log = ranks[0] == -1
        should_log = should_log or (my_rank in set(ranks))
    if should_log:
        final_message = "[Rank {}] {}".format(my_rank, message)
        logger.log(level, final_message)


def print_rank(message, rank=0, debug=False, force=False):
    if get_rank() == rank and (debug or force):
        logger.info(message)
