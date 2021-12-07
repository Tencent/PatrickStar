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

import torch

from patrickstar.core import ChunkList, ChunkTensorIndex, ParamType
from patrickstar.utils import logger, get_rank
from patrickstar.core.memory_cache import MemoryCache
from typing import Optional


class FP16ChunkWriteBuffer(object):
    r"""A buffer for copy the param.

    At the end of the CPU Adam, we need to copy the updated fp32 params
    back to the fp16 params for the next iteration of training.
    And because the params are organized in chunks, we can optimize the copy and cast
    by doing it at the granularity of chunk.
    This class is for doing the above copy and cast optimization.
    """

    def __init__(
        self,
        chunk_list: ChunkList,
        chunk_tensor_index: ChunkTensorIndex,
        chunk_size: int,
        mem_cache: Optional[MemoryCache] = None,
    ):
        """
        Args:
            chunk_list: :class:`ChunkList`.
            chunk_tensor_index: :class:`ChunkTensorIndex`.
            chunk_size: `int`.
        """
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index
        self.cached_src_chunk_id = None
        self.cached_target_chunk_id = None
        # NOTE() We found that doing a two stage copy, 1) CPU fp32 -> GPU fp32,
        # 2) GPU fp32 -> GPU fp16 is faster than one single copy_. And the
        # gpu_fp32_buff member is the itermediate buffer.

        self.with_mem_cache = mem_cache is not None
        if self.with_mem_cache:
            self.memory_cache = mem_cache
            self.gpu_fp32_buff = self.memory_cache.pop_or_allocate(
                torch.device(f"cuda:{torch.cuda.current_device()}"),
                chunk_size,
                torch.float,
                False,
            )
        else:
            self.gpu_fp32_buff = torch.zeros(
                chunk_size,
                dtype=torch.float,
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
            )

    def write_from_cache(self, target_param, src_param):
        r"""Write the value of `target_param` to `src_param` with casting.

        We assume the order of the `src_param` and `target_param` coming in
        are the same as the order they reside in chunks. Therefore, we
        will copy the chunk of them only if `src_param` is the last param
        in its chunk.
        Note that the last chunk will be copied in `reset`.

        Args:
            target_param: A :class:`torch.nn.Parameter`. The fp16 param to copy to.
            src_param: A :class:`torch.nn.Parameter`. The fp32 param to copy from.
        """
        # Torch params are of fp32 all the time, so we don't need to copy them.
        assert src_param.ps_attr.param_type == ParamType.CHUNK_BASED
        src_info = self.chunk_tensor_index.get_tensor_info(src_param.ps_attr.data_id())
        target_info = self.chunk_tensor_index.get_tensor_info(
            target_param.ps_attr.data_id()
        )

        if (
            self.cached_src_chunk_id is not None
            and src_info.chunk_id != self.cached_src_chunk_id
        ):
            # TODO(jiaruifang) Optimize CPU -> GPU copy.
            target_device = self.chunk_list[self.cached_target_chunk_id].payload.device
            src_device = self.chunk_list[self.cached_src_chunk_id].payload.device
            logger.debug(
                f"Write chunk {self.cached_src_chunk_id} -> {self.cached_target_chunk_id}, "
                f"{src_device} -> {target_device}"
            )
            if target_device.type == "cuda" and src_device.type == "cpu":
                self.gpu_fp32_buff.copy_(
                    self.chunk_list[self.cached_src_chunk_id].payload
                )
                self.chunk_list[self.cached_target_chunk_id].payload.copy_(
                    self.gpu_fp32_buff
                )
            else:
                self.chunk_list[self.cached_target_chunk_id].payload.copy_(
                    self.chunk_list[self.cached_src_chunk_id].payload
                )
        self.cached_src_chunk_id = src_info.chunk_id
        self.cached_target_chunk_id = target_info.chunk_id

    def reset(self):
        r"""Reset the chunk buffer.

        During reset, we will copy the last chunk from fp32 to fp16.
        """
        if self.cached_src_chunk_id is None:
            return
        global_rank = get_rank()
        logger.info(
            f"global_rank {global_rank} finally, write chunk {self.cached_target_chunk_id}"
        )
        # It's possible that the chunk is empty (no payload), e.g. the process only possesses
        # a large torch based embedding layer.
        if self.chunk_list[self.cached_src_chunk_id] is not None:
            self.chunk_list[self.cached_target_chunk_id].payload.copy_(
                self.chunk_list[self.cached_src_chunk_id].payload
            )
        self.cached_src_chunk_id = None
        self.cached_target_chunk_id = None
        if self.with_mem_cache:
            self.memory_cache.push(self.gpu_fp32_buff)
            self.gpu_fp32_buff = None


class FP32ChunkReadBuffer(object):
    r"""Read param from chunk.

    During Adam, we will need to access the fp16 chunks for the
    gradients and sometimes copy them to GPU.
    As they are organized in chunks, we will move them by chunks.
    This class is for such optimization.
    """

    def __init__(
        self,
        chunk_list: ChunkList,
        chunk_tensor_index: ChunkTensorIndex,
        chunk_size: int,
        margin_chunk_num_for_gpu_adam: int,
        mem_cache: Optional[MemoryCache] = None,
    ):
        """
        Args:
            chunk_list: :class:`ChunkList`.
            chunk_tensor_index: :class:`ChunkTensorIndex`.
            chunk_size: `int`.
            margin_chunk_num_for_gpu_adam: `int`. the number of GPU chunks for Adam state.
        """
        self.chunk_list = chunk_list
        self.chunk_tensor_index = chunk_tensor_index
        self.cpu_payload = torch.empty(
            chunk_size, dtype=torch.half, device=torch.device("cpu:0"), pin_memory=True
        )
        self.local_rank = chunk_list.local_rank

        self.with_mem_cache = mem_cache is not None
        if self.with_mem_cache:
            self.memory_cache = mem_cache

        self.gpu_payload = None
        if margin_chunk_num_for_gpu_adam > 0:
            # When `margin_chunk_num_for_gpu_adam` > 0, it means there will be optimizer
            # state resides on GPU. So we need to allocate a GPU buffer for those.
            gpu_device = torch.device(f"cuda:{self.local_rank}")

            if self.with_mem_cache:
                self.gpu_payload = self.memory_cache.pop_or_allocate(
                    gpu_device, chunk_size, torch.half, False
                )
            else:
                logger.info(
                    f"Allocate fp32 Chunk Buffer of size {chunk_size / 1e6} MB on CPU."
                )
                self.gpu_payload = torch.empty(
                    chunk_size, dtype=torch.half, device=gpu_device
                )
            logger.info(
                f"Allocate fp32 Chunk Buffer of size {chunk_size / 1e6} MB on {gpu_device}."
            )
        self.cached_chunk_id = None
        self.margin_chunk_num_for_gpu_adam = margin_chunk_num_for_gpu_adam
        self.cached_chunk_num = 0
        self.ret_payload = None

    def access_from_cache(self, param) -> torch.Tensor:
        r"""Access the underlying data of the param.

        We assume the order of the `param` coming in in the same as the order
        they reside in chunks. Therefore, we will copy the chunk when `param`
        is the first one in the chunk.
        As target device may be CPU or GPU, we will need 2 kinds of payload to
        store the buffer.

        Args:
            param: :class:`torch.nn.Parameter`. The param of dtype fp16 to access.
        """
        if param.ps_attr.param_type == ParamType.TORCH_BASED:
            # grad of torch params are stored in param.grad.
            return param.grad
        else:
            info = self.chunk_tensor_index.get_tensor_info(param.ps_attr.data_id())

            # Trigger updation of cached chunk when param is the first tensor of
            # its chunk.
            if info.start_offset == 0:
                self.cached_chunk_num += 1
                if self.cached_chunk_num < self.margin_chunk_num_for_gpu_adam:
                    target_device = torch.device(f"cuda:{self.local_rank}")
                else:
                    target_device = torch.device("cpu:0")

                chunk_payload = self.chunk_list[info.chunk_id].payload
                if target_device.type == "cuda":
                    self.gpu_payload.copy_(chunk_payload)
                    self.ret_payload = self.gpu_payload
                elif target_device.type == "cpu":
                    self.cpu_payload.copy_(chunk_payload)
                    self.ret_payload = self.cpu_payload
                logger.debug(
                    f"Read chunk {self.cached_chunk_id} to cache "
                    f"{chunk_payload.device} -> {target_device}"
                )
                self.cached_chunk_id = info.chunk_id
            else:
                assert info.chunk_id == self.cached_chunk_id
            return self.ret_payload.narrow(0, info.start_offset, info.numel)

    def reset(self):
        self.cached_chunk_num = 0
        self.ret_payload = None
        self.cached_chunk_id = None
        if self.with_mem_cache:
            if self.gpu_payload is not None:
                self.memory_cache.push(self.gpu_payload)
                self.gpu_payload = None
