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

from patrickstar.core.const import TrainingStage


class Metronome:
    """
    A metronome for memory stats sampling.
    Use two indicators to tell us where the training is now
    One is moment, indicates the moment of one iteration.
    The other is training stage, indicates FWD/BWD/ADAM and is this iteration is
    a warmup iteration.

    It also contain the training stage information.
    """

    def __init__(self):
        self.moment = 0
        self.total_moment = None
        self.training_stage = TrainingStage.UNSTART
        self.is_warmup = False

    def tiktac(self):
        """
        The function should be called right before and after computing of an operator.
        """
        self.moment += 1

    def reset(self):
        """
        The function is called after a trainig iteration is finished.
        """
        self.total_moment = self.moment
        self.moment = 0

    def next_moment(self):
        assert self.total_moment is not None
        return min(self.total_moment, self.moment + 1) % self.total_moment

    def prev_moment(self):
        assert self.total_moment is not None
        return max(0, self.moment - 1) % self.total_moment
