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

import torch
import torch.nn as nn
from deepspeed_helper.global_vars import get_args
from utils import logger


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.vocab_size = 10
        self.max_position_embeddings = 512
        self.hidden_size = hidden_dim
        self.type_vocab_size = 2

        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings,
                                                self.hidden_size)
        self.token_type_embeddings = nn.Embedding(self.type_vocab_size,
                                                  self.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(self.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(self, "position_embedding_type",
                                               "absolute")

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:
                                             seq_length +
                                             past_key_values_length]  #.cuda()

        # TODO(jiaruifang)为了PS adam修改的，如果模型初始化在cpu上，并没有机制把ids显式移动到
        # cuda设备上。多机情况尚未考虑。
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long,
                device=self.position_ids.device)  #.cuda()

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        # start computing

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class _CopyToDatalParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        return input_.to(torch.device('cpu:0'))

    @staticmethod
    def forward(ctx, input_):
        logger.info('copy input to cpu')
        return input_.to(torch.device('cpu:0'))

    @staticmethod
    def backward(ctx, grad_output):
        logger.info('copy grad_output to cuda')
        return grad_output.to(torch.device('cuda:0'))


class _CopyFromDatalParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        return input_.to(torch.device('cuda:0'))

    @staticmethod
    def forward(ctx, input_):
        logger.info('copy input to cuda')
        return input_.to(torch.device('cuda:0'))

    @staticmethod
    def backward(ctx, grad_output):
        logger.info('copy grad_output to cpu')
        return grad_output.to(torch.device('cpu:0'))


def copy_to_data_parallel_region(input_):
    return _CopyToDatalParallelRegion.apply(input_)


def gather_from_data_parallel_region(input_):
    return _CopyFromDatalParallelRegion.apply(input_)


class ParallelBertEmbeddings(nn.Module):
    def __init__(self, hidden_dim):
        super(ParallelBertEmbeddings, self).__init__()
        self.bert_embedding = BertEmbeddings(hidden_dim)

    def forward(self,
                input_ids=None,
                token_type_ids=None,
                position_ids=None,
                inputs_embeds=None,
                past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        assert inputs_embeds is None
        assert token_type_ids is None
        assert position_ids is None

        input_ids_parallel = copy_to_data_parallel_region(input_ids)

        args = get_args()
        if 0 == args.local_rank:
            output_parallel = self.bert_embedding(input_ids_parallel,
                                                  token_type_ids, position_ids)

        output = gather_from_data_parallel_region(output_parallel)
        return output
