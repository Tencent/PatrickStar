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

import torch
from rich.logging import RichHandler


class LoggerFactory:
    @staticmethod
    def create_logger(name, level=logging.WARNING):
        """create a logger
        Args:
            name (str): name of the logger
            level: level of logger
        Raises:
            ValueError is name is None
        """

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

        logger_ = logging.getLogger(name)
        logger_.setLevel(level)
        logger_.propagate = False
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setFormatter(formatter)
        logger_.addHandler(RichHandler())
        return logger_


logger = LoggerFactory.create_logger(name="PatrickStar", level=logging.WARNING)


def log_dist(message, ranks=[0], level=logging.INFO):
    if not torch.distributed.is_initialized():
        return
    rank = torch.distributed.get_rank()
    if rank in ranks:
        logger.log(level, "[Rank {}] {}".format(rank, message))
