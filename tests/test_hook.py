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

from utils.hook import setup_hybrid_ps_hooks
from tests.simple_net import SimpleModel, get_data_loader
import logging
import torch
from ops import CPUAdam
from client import HybridPSClient
from manager import HybridPSManager


def test():
    hidden_dim = 4
    device = torch.device('cuda:0')

    model = SimpleModel(hidden_dim, empty_grad=False)
    model.cuda()

    data_loader = get_data_loader(model=model,
                                  total_samples=1000,
                                  hidden_dim=hidden_dim,
                                  device=device,
                                  data_type=torch.float)

    loss_res = []
    logging.warning('before register model')
    client = HybridPSClient(gpu_index=0, default_chunk_size=20)
    optimizer = CPUAdam(client, model.parameters(), lr=0.001)
    client.register_module(model)
    logging.warning('after register model')
    setup_hybrid_ps_hooks(model, client)

    for n, batch in enumerate(data_loader):
        logging.warning(f'before fwd step {n}')

        loss = model(batch[0], batch[1])

        logging.warning(f'after fwd step {n}')
        logging.warning(f"LOSS: {loss.item()}")
        loss_res.append(loss.item())

        optimizer.zero_grad()
        loss.backward()

        logging.warning(f'before step {n}')
        optimizer.step()
        logging.warning(f'end step {n}')

        client.release_all_grad()
        if n == 5: break


if __name__ == "__main__":
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.WARNING)
    torch.manual_seed(0)
    manager = HybridPSManager()
    # 4 layer每层20个elem(20*4 bytes)，最少360个elem (360*4 bytes)内存
    # M, V, P FP32 240
    manager.init([40 * 4] * 1, [280 * 4])
    torch.manual_seed(0)
    test()

    print(manager.gpu_mem_usage_curve)
    print(manager.cpu_mem_usage_curve)
