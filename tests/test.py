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

from manager import HybridPSManager
from client import HybridPSClient
import torch
import torch.distributed as dist
from common import distributed_test
import time
import logging

manager = HybridPSManager()


@distributed_test(world_size=1)
def test_client():
    world_size = dist.get_world_size()
    manager.init(gpu_info=[32 * 2] * world_size, cpu_info=[64 * 2])
    print("is init manager", HybridPSManager().is_init())
    local_rank = dist.get_rank()

    # 申请两个tensor
    param1 = torch.randn(40,
                         device=torch.cuda.current_device()
                         if torch.cuda.is_available() else torch.device('cpu'))
    param2 = torch.randn(15,
                         device=torch.cuda.current_device()
                         if torch.cuda.is_available() else torch.device('cpu'))

    # 用一个HybridPSClient来管理这两个tensor
    # GPU Chunk 40, 20
    client = HybridPSClient(gpu_index=local_rank, default_chunk_size=20)

    client.register_param(param1)
    client.register_param(param2)

    assert param1.device == torch.device('cpu')
    assert param2.device == torch.device('cpu')

    # 申请第三个tensor，此时cpu内存不足，被放在gpu上
    # GPU Chunk 40, 20 CPU Chunk 20
    param3 = torch.randn(20, device=torch.device('cpu'))
    client.register_param(param3)
    assert param3.device == torch.device('cuda:0')

    # 申请第四个tensor, 需要chunk size=20大小, GPU没有空间了，会跑出异常
    except_flag = False
    try:
        param4, _ = client.new_tensor((1, 5))
    except:
        except_flag = True
    assert (except_flag)


world_size = 2


def test_mgr_dist():
    @distributed_test(world_size=world_size)
    def test_dist_init():
        assert dist.is_initialized()
        assert dist.get_world_size() == world_size
        assert dist.get_rank() < 2
        print("pass test_init")

    #测试mgr正确更新
    def test_mgr_update():
        # 在两个进程上使用HybridPSClient，测试manager效果
        manager = HybridPSManager()
        manager.reset([32, 32], [64])

        @distributed_test(world_size=world_size)
        def test_add():
            local_rank = dist.get_rank()
            manager.add("cuda", local_rank, (local_rank + 1) * 10)
            manager.add("cuda", local_rank, (local_rank + 1) * 22)

        @distributed_test(world_size=world_size)
        def test_delete():
            local_rank = dist.get_rank()
            if local_rank == 0:
                manager.delete("cuda", local_rank, 10)

        test_add()
        assert (manager.used_mem("cuda", 0) == 32)
        assert (manager.used_mem("cuda", 1) == 64)
        assert (manager.used_mem("cpu", 0) == 0)
        time.sleep(3)
        test_delete()
        assert (manager.used_mem("cuda", 0) == 22)
        assert (manager.used_mem("cuda", 1) == 64)
        print("pass test_mgr_update")

    test_dist_init()
    time.sleep(2)
    test_mgr_update()


def test_migrate():
    @distributed_test(world_size=1)
    def test_access():
        if not torch.cuda.is_available():
            print('cuda is not available in test_access')

        compute_device = torch.device('cuda:0')
        local_rank = dist.get_rank()
        manager = HybridPSManager()
        manager.reset(gpu_info=[80], cpu_info=[200])

        # 申请两个tensor, 他们放在一个chunk中，计算设备在cuda上
        param1 = torch.randn(20, device=torch.device('cuda:0'))
        param2 = torch.randn(20, device=torch.device('cuda:0'))

        # 交给HybridPS管理，会先被分在cpu上
        client = HybridPSClient(gpu_index=local_rank, default_chunk_size=40)
        client.register_param(param1, compute_device)
        client.register_param(param2, compute_device)

        # print(param1.ps_data_id, )
        assert client.is_ps_param(
            param1) and param1.ps_data_id == 0 and param1.ps_grad_id == 1
        assert param1.device.type == 'cpu'
        assert param2.device.type == 'cpu'

        # 访问param
        client.access_data(param1, compute_device)
        assert param1.device.type == 'cuda'
        client.access_data(param2, compute_device)
        assert param2.device.type == 'cuda'
        # assert param1.ps_data_chunk_id == param2.ps_data_chunk_id

        # chunk 2 放在gpu上, gpu空间只能容纳一个chunk
        param3 = torch.randn(20, device=torch.device('cuda:0'))
        client.register_param(param3)
        client.chunk_move(param3.ps_data_chunk_id, torch.device('cuda:0'))

        assert param3.device.type == 'cuda'

        client.visit()

        client.release_data(param1)
        client.release_data(param2)
        client.visit()
        print("[PASS] test_migrate - test_access")

    @distributed_test(world_size=1)
    def test_chunk_to_move_out_for_room_making():
        if not torch.cuda.is_available():
            print('cuda is not available in test_access')
        compute_device = torch.device('cuda:0')

        local_rank = dist.get_rank()
        manager = HybridPSManager()
        manager.reset(gpu_info=[80], cpu_info=[200])

        # 申请两个tensor, 他们放在一个chunk中，计算设备在cuda上
        param1 = torch.nn.Parameter(
            torch.randn(20, device=torch.device('cuda:0')))
        param2 = torch.nn.Parameter(
            torch.randn(20, device=torch.device('cuda:0')))

        # 交给HybridPS管理，会先被分在cpu上, 占据了2个chunk
        client = HybridPSClient(gpu_index=local_rank, default_chunk_size=40)
        client.register_param(param1)
        client.register_param(param2)

        client.visit()

        # 访问param，两个chunk被move到gpu上
        client.access_data(param1, compute_device)
        client.access_grad(param1, compute_device)
        client.access_data(param2, compute_device)
        client.access_grad(param2, compute_device)

        # gpu上空出一个chunk
        client.release_data(param1)
        client.release_grad(param1)

        param3 = torch.randn(24, device=torch.device('cuda:0'))
        client.register_param(param3)
        client.access_data(param3, compute_device)
        #需要移动gpu chunk 1 from gpu -> cpu
        client.visit()
        print("[PASS] test_migrate - test_chunk_to_move_out_for_room_making")

    # test_access()
    test_chunk_to_move_out_for_room_making()


if __name__ == "__main__":
    # test_client()
    # time.sleep(3)
    # test_mgr_dist()
    # time.sleep(3)
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    test_migrate()