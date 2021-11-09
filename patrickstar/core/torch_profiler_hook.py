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

from patrickstar.core.hook import (
    PreBackwardFunction,
    PostBackwardFunction,
    _apply_to_tensors_only,
)
from patrickstar.utils import get_sys_memory_used, logger

from patrickstar.profiler import profiler
from patrickstar.utils.memory_monitor import max_mem_usage_period


def _cur_mem_usage():
    """
    A function used to sample memory usage at the moment
    before and after an operator sharted and finished.
    """
    dev = torch.device(f"cuda:{torch.cuda.current_device()}")
    gpu_mem_used = get_sys_memory_used(dev)
    return gpu_mem_used


def _record_mem_stats():
    """
    Record memory statistics at this moment for the profiler.
    """
    mem_cur_mon = _cur_mem_usage()
    # In case of an operator running too short,
    # the sampler dose not capture any information.
    # we add mem of cur mom as the default memory usage of the period.
    max_mem_between_cur_prev = max(max_mem_usage_period(), mem_cur_mon)

    profiler.gpu_memory_used.append(
        (None, time.time(), max_mem_between_cur_prev, mem_cur_mon)
    )


def _register_hooks_recursively(module, name=""):
    r"""Register hook in post order traverse."""

    for child_name, child in module.named_children():
        logger.debug(f"{child.__class__.__name__}")
        _register_hooks_recursively(child, name + child_name)

    # Early return on modules with no parameters or buffers that
    # are not in their children.
    if (
        len(list(module.named_parameters(recurse=False))) == 0
        and len(list(module.named_buffers(recurse=False))) == 0
    ):
        return

    def _pre_post_forward_module_hook(module, *args):
        _record_mem_stats()

    # The hook can modify the output
    def _pre_backward_module_hook(module, inputs, output):
        def _run_before_backward_function(sub_module):
            _record_mem_stats()

        return _apply_to_tensors_only(
            module, PreBackwardFunction, _run_before_backward_function, output
        )

    def _post_backward_module_hook(module, inputs):
        def _run_after_backward_function(sub_module):
            _record_mem_stats()

        return _apply_to_tensors_only(
            module, PostBackwardFunction, _run_after_backward_function, inputs
        )

    module.register_forward_pre_hook(_pre_post_forward_module_hook)
    module.register_forward_hook(_pre_post_forward_module_hook)

    module.register_forward_hook(_pre_backward_module_hook)
    module.register_forward_pre_hook(_post_backward_module_hook)


def register_torch_profiler_hook(module):
    """
    Collect activation statistis during training.
    """
    _register_hooks_recursively(module)
