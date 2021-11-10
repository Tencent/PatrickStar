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

from .training_stage_mgr import TrainingStageMgr


class Metronome(object):
    """
    A metronome for memory stats sampling.
    Use two indicators to tell us where the training is now
    One is moment, indicates the moment of one iteration.
    The other is training stage, indicates FWD/BWD/ADAM and is this iteration is
    a warmup iteration.

    It also contain the training stage information.
    """

    def __init__(self):
        self._moment = 0
        self._total_moment = None
        self.training_stage_mgr = TrainingStageMgr()

    def set_training_phase(self, phase):
        self.training_stage_mgr.training_phase = phase

    def set_warmup(self, flag):
        self.training_stage_mgr.is_warmup = flag

    def is_warmup(self):
        return self.training_stage_mgr.is_warmup

    def training_stage(self):
        return self.training_stage_mgr.training_phase

    def get_total_mom(self):
        assert self._total_moment is not None, "Don not use get_total during warmup"
        return self._total_moment

    def tiktac(self):
        """
        The function should be called right before and after computing of an operator.
        """
        self._moment += 1

    def moment(self):
        return self._moment

    def reset(self):
        """
        The function is called after a trainig iteration is finished.
        """
        self._total_moment = self._moment
        self._moment = 0

    def next_moment(self):
        assert self._total_moment is not None
        return min(self._total_moment, self._moment + 1) % self._total_moment

    def prev_moment(self):
        assert self._total_moment is not None
        return max(0, self._moment - 1) % self._total_moment
