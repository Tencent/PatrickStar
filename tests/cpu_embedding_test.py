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
from patrickstar.ops import CpuBertEmbeddings
from patrickstar.utils import logger
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEmbeddings
from common import distributed_test
import torch


class TestClientAccess(unittest.TestCase):
    def setUp(self):
        pass

    @distributed_test(world_size=[1])
    def test_cpu_embedding_layer(self):
        cfg = BertConfig()
        cfg.hidden_dropout_prob = 0
        test_device = torch.device('cuda:0')
        seq_len = 10
        torch.manual_seed(0)
        input_ids = torch.randint(low=0,
                                  high=cfg.vocab_size - 1,
                                  size=(1, seq_len),
                                  dtype=torch.long,
                                  device=test_device)

        torch.manual_seed(0)
        cpu_embedding = CpuBertEmbeddings(cfg)
        torch.manual_seed(0)
        bert_embedding = BertEmbeddings(cfg)
        bert_embedding = bert_embedding.to(test_device)

        cpu_embedding.LayerNorm.to(test_device)
        cpu_embedding.dropout.to(test_device)
        res = cpu_embedding(input_ids)

        # torch
        torch_res = bert_embedding(input_ids)
        self.assertLess(torch.max(torch_res.cpu() - res.cpu()), 1e-4)
        print("cpu embedding check OK")

    @distributed_test(world_size=[2], backend='gloo', use_fake_dist=True)
    def test_p2p_api(self):
        test_device = torch.device('cpu:0')
        input_ids = torch.randint(low=0,
                                  high=10 - 1,
                                  size=(1, 20),
                                  dtype=torch.long,
                                  device=test_device)
        rank = torch.distributed.get_rank()
        if rank == 0:
            torch.distributed.send(input_ids, dst=1)
        else:
            torch.distributed.recv(input_ids, src=0)

    @distributed_test(world_size=[2], backend='gloo', use_fake_dist=True)
    def test_send_ids_to_rank0(self):
        from patrickstar.ops.cpu_embedding import send_ids_to_rank0
        seq_len = 20
        test_device = torch.device('cpu:0')
        input_ids = torch.randint(low=0,
                                  high=10 - 1,
                                  size=(1, seq_len),
                                  dtype=torch.long,
                                  device=test_device)
        rank = torch.distributed.get_rank()
        gathered_input_ids = send_ids_to_rank0(input_ids)
        if rank == 0:
            self.assertTrue(
                gathered_input_ids.shape[0] == 2,
                f"the batch dim of gathered id should be 2, now {gathered_input_ids.shape[0]}"
            )

    @distributed_test(world_size=[2], backend='gloo', use_fake_dist=True)
    def test_collect_act_from_rank0(self):
        from patrickstar.ops.cpu_embedding import collect_act_from_rank0
        seq_len = 20
        test_device = torch.device(f'cuda:{torch.cuda.current_device()}')

        global_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        if global_rank == 0:
            input_ids = torch.randn(4 * world_size, 10, device=test_device)
        else:
            input_ids = torch.randn(4, 10, device=test_device)
        gathered_input_ids = collect_act_from_rank0(input_ids)

        if global_rank == 0:
            self.assertTrue(gathered_input_ids.shape[0] == 4,
                            "the batch dim of gathered id should be 2")


if __name__ == "__main__":
    unittest.main()
