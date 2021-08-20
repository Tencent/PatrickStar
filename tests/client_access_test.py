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
from patrickstar.core import PatrickStarClient, ChunkList, PSTensorStatus, ChunkTensorIndex, PSChunkStatus, register_param, ChunkListType
import logging
import torch
from common import distributed_test
from patrickstar.deepspeed_helper.global_vars import set_global_variables, get_args


class TestClientAccess(unittest.TestCase):
    def setUp(self):
        self.default_chunk_size = 40
        logging.info('SetUp finished')

    @distributed_test(world_size=[1])
    def test_append_tensor(self):
        self.client = PatrickStarClient(
            rank=0, default_chunk_size=self.default_chunk_size)

        self.compute_device = torch.device('cpu:0')
        # self.compute_device = torch.device(
        #     f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else torch.device('cpu:0')

        param_size_list = [10, 11, 12, 13]

        param_list = []
        param_payload_ref_list = []
        for idx, psize in enumerate(param_size_list):
            param = torch.nn.Parameter(torch.rand(psize))
            param_list.append(param)
            param_payload_ref_list.append(param.data.clone())
            self.client.append_tensor(param,
                                      ChunkListType.PARAM_FP32, f"param{idx}")

            # TODO(jiaruifang) access_data叫try_fetch_data更恰当，并没有返回一个tensor
            self.client.access_data(param, torch.device('cpu:0'))
            real_payload = param.ps_attr.access_tensor()
            real_payload.copy_(param.data)
            param.data = torch.tensor([])
            self.client.release(param)

        self.client.chunk_tensor_index.visit_chunks(self.client.chunk_list)
        for param, payload_ref in zip(param_list, param_payload_ref_list):
            self.client.access_data(param, torch.device('cpu:0'))
            real_payload = param.ps_attr.access_tensor()
            self.assertEqual(torch.max(real_payload - payload_ref), 0)
            self.client.release(param)


if __name__ == "__main__":
    set_global_variables()
    logging.basicConfig(
        format=
        '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.DEBUG)
    unittest.main()
