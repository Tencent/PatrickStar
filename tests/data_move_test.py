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

import unittest
from client import HybridPSClient, ChunkList, PSTensorStatus, AccessType
import logging
import torch
from manager import HybridPSManager
from utils import see_memory_usage
import contexttimer
import time


class TestAccess(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 20
        self.client = HybridPSClient(
            gpu_index=0, default_chunk_size=self.default_chunk_size)
        self.manager = HybridPSManager()
        self.manager.init([32, 32], [1024])
        self.compute_device = torch.device(
            'cuda') if torch.cuda.is_available() else torch.device('cpu')
        logging.info('SetUp finished')

    def _release_cpu_allocate_gpu(self, size):
        elapsed = 0
        for i in range(10):
            # cpu_buff = torch.ones(size, dtype=torch.float, device=torch.device('cpu:0'))

            start_time = time.time()
            torch.cuda.synchronize()
            # 分配cpu
            cpu_buff = torch.ones(size,
                                  dtype=torch.float,
                                  device=torch.device('cpu:0'))
            del cpu_buff
            torch.cuda.synchronize()
            #
            elapsed += time.time() - start_time
            # del gpu_buff

        BWD = size * 4 / (elapsed / 10) / 1e9
        logging.info(
            f'Release/Allocate size {size * 4/1024} KB, bandwidth {BWD} GB/s')

    def _copy_bandwidth_benchmark(self, size, src_device, target_device):
        src_buff = torch.ones(size, dtype=torch.float, device=src_device)
        dest_buff = torch.zeros(size, dtype=torch.float, device=target_device)

        with contexttimer.Timer() as t:
            dest_buff.copy_(src_buff)
        BWD = size * 4 / t.elapsed / 1e9
        logging.info(
            f'Copy {src_device} to {target_device}, size {size * 4/1024} KB, bandwidth {BWD} GB/s'
        )

    def _inline_copy_bandwidth_benchmark(self, size, src_device,
                                         target_device):
        buff = torch.zeros(size, dtype=torch.float, device=src_device)
        with contexttimer.Timer() as t:
            buff.to(target_device)
        BWD = size * 4 / t.elapsed / 1e9
        logging.info(
            f'Copy {src_device} to {target_device}, size {size * 4/1024} KB, bandwidth {BWD} GB/s'
        )

    def _pinned_copy_bandwidth_benchmark(self, size, src_device,
                                         target_device):
        src_buff = torch.ones(size, dtype=torch.float, device=src_device)
        if src_device.type == 'cpu':
            src_buff = src_buff.pin_memory()
        dest_buff = torch.zeros(size, dtype=torch.float, device=target_device)
        if target_device.type == 'cpu':
            dest_buff = dest_buff.pin_memory()

        with contexttimer.Timer() as t:
            dest_buff.copy_(src_buff)
        BWD = size * 4 / t.elapsed / 1e9
        logging.info(
            f'Pined-Copy {src_device} to {target_device}, size {size * 4/1024} KB, bandwidth {BWD} GB/s'
        )

    def _async_copy_bandwidth_benchmark(self, size, src_device, target_device):
        src_buff = torch.ones(size, dtype=torch.float, device=src_device)
        if src_device.type == 'cpu':
            src_buff = src_buff.pin_memory()
        dest_buff = torch.zeros(size, dtype=torch.float, device=target_device)
        if target_device.type == 'cpu':
            dest_buff = dest_buff.pin_memory()

        copy_grad_stream = torch.cuda.Stream()
        with contexttimer.Timer() as t:
            with torch.cuda.stream(copy_grad_stream):
                dest_buff.copy_(src_buff, non_blocking=True)
            copy_grad_stream.synchronize()
            torch.cuda.synchronize()
        res = torch.sum(gpu_buff)
        assert res.item() == torch.sum(cpu_buff).item()
        BWD = size * 4 / t.elapsed / 1e9
        logging.info(
            f'Async-Copy {src_device} to {target_device}, size {size * 4/1024} KB, bandwidth {BWD} GB/s'
        )

    def _async_copy_inline_bandwidth_benchmark(self, size, src_device,
                                               target_device):
        """
        使用inline方式拷贝，non_blocking=True只有当src_device在pin_memory CPU上时有效
        """
        src_buff = torch.ones(size, dtype=torch.float, device=src_device)
        if src_device.type == 'cpu':
            src_buff = src_buff.pin_memory()
        dest_buff = torch.zeros(size, dtype=torch.float, device=target_device)
        if target_device.type == 'cpu':
            dest_buff = dest_buff.pin_memory()

        copy_grad_stream = torch.cuda.Stream()
        with contexttimer.Timer() as t:
            with torch.cuda.stream(copy_grad_stream):
                src_buff.to(target_device, non_blocking=True)
            copy_grad_stream.synchronize()
            # torch.cuda.synchronize()
        res = torch.sum(src_buff)
        assert res.item() == size
        BWD = size * 4 / t.elapsed / 1e9
        logging.info(
            f'Async-Copy Inline {src_device} to {target_device}, size {size * 4/1024} KB, bandwidth {BWD} GB/s'
        )

    def test_bandwidth(self):
        for size in [
                1024, 32 * 1024, 64 * 1024, 128 * 1024, 512 * 1024,
                1024 * 1024, 1024 * 1024 * 8, 1024 * 1024 * 13,
                1024 * 1024 * 32
        ]:
            self._release_cpu_allocate_gpu(size)
            # self._copy_bandwidth_benchmark(size, torch.device('cpu'),
            #                                torch.device('cuda'))
            # self._copy_bandwidth_benchmark(size, torch.device('cuda'),
            #                                torch.device('cpu'))
            # self._copy_bandwidth_benchmark(size, torch.device('cuda'),
            #                                torch.device('cuda'))
            # self._copy_bandwidth_benchmark(size, torch.device('cpu'),
            #                                torch.device('cpu'))

            # self._inline_copy_bandwidth_benchmark(size, torch.device('cpu'),
            #                                       torch.device('cuda'))
            # self._inline_copy_bandwidth_benchmark(size, torch.device('cuda'),
            #                                       torch.device('cpu'))

            self._pinned_copy_bandwidth_benchmark(size, torch.device('cpu'),
                                                  torch.device('cuda'))
            self._pinned_copy_bandwidth_benchmark(size, torch.device('cuda'),
                                                  torch.device('cpu'))

            self._async_copy_inline_bandwidth_benchmark(
                size, torch.device('cpu'), torch.device('cuda'))
            self._async_copy_inline_bandwidth_benchmark(
                size, torch.device('cuda'), torch.device('cpu'))
            logging.info(f'==========')


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
