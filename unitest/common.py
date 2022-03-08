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

import os
import time

import torch
from torch.multiprocessing import Process

# Worker timeout *after* the first worker has completed.
UNIT_WORKER_TIMEOUT = 120


def distributed_test(world_size=2, backend="nccl", use_fake_dist=False):
    r"""A decorator for executing a function (e.g., a unit test) in a distributed manner.

    This decorator manages the spawning and joining of processes, initialization of
    torch.distributed, and catching of errors.

    Usage example:
        @distributed_test(worker_size=[2,3])
        def my_test():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            assert(rank < world_size)

    Args:
        world_size (int or list): number of ranks to spawn. Can be a list to spawn
        multiple tests.
    """

    def dist_wrap(run_func):
        """Second-level decorator for dist_test. This actually wraps the function."""

        def dist_init(local_rank, num_procs, *func_args, **func_kwargs):
            """Initialize torch.distributed and execute the user function."""
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            os.environ["MASTER_PORT"] = "29503"
            os.environ["LOCAL_RANK"] = str(local_rank)
            # NOTE: unit tests don't support multi-node so local_rank == global rank
            os.environ["RANK"] = str(local_rank)
            os.environ["WORLD_SIZE"] = str(num_procs)

            torch.distributed.init_process_group(backend=backend)
            if torch.cuda.is_available():
                if use_fake_dist:
                    torch.cuda.set_device(0)
                else:
                    torch.cuda.set_device(local_rank)
            run_func(*func_args, **func_kwargs)

        def dist_launcher(num_procs, *func_args, **func_kwargs):
            r"""Launch processes and gracefully handle failures."""

            # Spawn all workers on subprocesses.
            processes = []
            for local_rank in range(num_procs):
                p = Process(
                    target=dist_init,
                    args=(local_rank, num_procs, *func_args),
                    kwargs=func_kwargs,
                )
                p.start()
                processes.append(p)

            # Now loop and wait for a test to complete. The spin-wait here isn't a big
            # deal because the number of processes will be O(#GPUs) << O(#CPUs).
            any_done = False
            while not any_done:
                for p in processes:
                    if not p.is_alive():
                        any_done = True
                        break

            # Wait for all other processes to complete
            for p in processes:
                p.join(UNIT_WORKER_TIMEOUT)

            failed = [(rank, p) for rank, p in enumerate(processes) if p.exitcode != 0]
            for _, p in failed:
                # If it still hasn't terminated, kill it because it hung.
                if p.exitcode is None:
                    p.terminate()
                if p.exitcode != 0:
                    p.terminate()

        def run_func_decorator(*func_args, **func_kwargs):
            r"""Entry point for @distributed_test()."""

            if isinstance(world_size, int):
                dist_launcher(world_size, *func_args, **func_kwargs)
            elif isinstance(world_size, list):
                for procs in world_size:
                    dist_launcher(procs, *func_args, **func_kwargs)
                    time.sleep(0.5)
            else:
                raise TypeError("world_size must be an integer or a list of integers.")

        return run_func_decorator

    return dist_wrap
