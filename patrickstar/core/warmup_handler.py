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

import functools
from patrickstar.utils import close_asyn_mem_monitor
from patrickstar.manager import PatrickStarManager


def adam_warmup_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        retval = func(*args, **kw)
        mgr = PatrickStarManager()
        if mgr.is_warmup_training():
            close_asyn_mem_monitor()
            # TODO(jiaruifang) rm mgr.warmup
            mgr.is_warmup = False
            mgr.metronome.training_stage.is_warmup = False
            print("----------------- WARMUP PHASE OVER -----------------")
        return retval

    return wrapper


class WarmupHandler(object):
    def __init__(self, client, warmup_steps: int = 1):
        """
        A handler to process warmup logic
        args:
            client: a patrickstar client.
            warmup_steps: run how many steps during training for warmup.
        """
        pass

    def process(step: int):
        """
        trigger warmup logic
        """
