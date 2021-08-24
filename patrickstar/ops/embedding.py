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
from patrickstar.deepspeed_helper.global_vars import get_args
from patrickstar.utils import logger
from patrickstar.core.parameter import is_torch_param, is_param_registered
from patrickstar.core import AccessType


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
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

        # 让临时生成的ids的设备和模型设备一致
        args = get_args()
        if args.use_fake_dist:
            device = torch.device('cuda:0')
        else:
            device = torch.device(f'cuda:{args.local_rank}')

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:
                                             seq_length +
                                             past_key_values_length].to(device)

        # TODO(jiaruifang)为了PS adam修改的，如果模型初始化在cpu上，并没有机制把ids显式移动到
        # cuda设备上。多机情况尚未考虑。
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long,
                device=self.position_ids.device).to(device)

        if inputs_embeds is None:
            assert input_ids.device == device, f"input_ids on {input_ids.device}, while computing device is {device}"
            inputs_embeds = self.word_embeddings(input_ids)

        # start computing
        print('token_type_ids device', token_type_ids.device,
              self.token_type_embeddings.weight.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertEmbeddingsWithoutLN(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, config):
        super().__init__()
        # TODO read them from config
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        self.hidden_size = config.hidden_size
        self.type_vocab_size = config.type_vocab_size

        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings,
                                                self.hidden_size)
        self.token_type_embeddings = nn.Embedding(self.type_vocab_size,
                                                  self.hidden_size)

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

        # 让临时生成的ids的设备和模型设备一致
        device = self.word_embeddings.weight.device

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length:
                                             seq_length +
                                             past_key_values_length].to(device)

        # TODO(jiaruifang)为了PS adam修改的，如果模型初始化在cpu上，并没有机制把ids显式移动到
        # cuda设备上。多机情况尚未考虑。
        if token_type_ids is None:
            token_type_ids = torch.zeros(
                input_shape, dtype=torch.long,
                device=self.position_ids.device).to(device)

        if inputs_embeds is None:
            assert input_ids.device == device, f"input_ids on {input_ids.device}, while computing device is {device}"
            inputs_embeds = self.word_embeddings(input_ids)

        # start computing
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        return embeddings


class _GatherActToRank0(torch.autograd.Function):
    """gather activations from other proc to proc 0"""
    @staticmethod
    def symbolic(graph, input_):
        args = get_args()
        return input_.to(torch.device('cpu:0'))

    @staticmethod
    def forward(ctx, input_):
        args = get_args()
        logger.info(
            f'Entrying CPU Emedding FWD, copy input to cpu and {input_.dtype}')
        return input_.to(torch.device('cpu:0'))

    @staticmethod
    def backward(ctx, grad_output):
        args = get_args()
        if args.use_fake_dist:
            target_device = torch.device('cuda:0')
        else:
            target_device = torch.device(f'cuda:{args.local_rank}')

        logger.info(
            'Entrying CPU Emedding BWD, copy grad_output to cuda, fp32->fp16')
        return grad_output.to(target_device)


class _BcastActFromRank0(torch.autograd.Function):
    """Pass the input to the model parallel region."""
    @staticmethod
    def symbolic(graph, input_):
        args = get_args()
        if args.use_fake_dist:
            target_device = torch.device('cuda:0')
        else:
            target_device = torch.device(f'cuda:{args.local_rank}')

        return input_.to(target_device).half()

    @staticmethod
    def forward(ctx, input_):
        args = get_args()
        if args.use_fake_dist:
            target_device = torch.device('cuda:0')
        else:
            target_device = torch.device(f'cuda:{args.local_rank}')

        logger.info(
            f'Entrying CPU Emedding BWD, copy grad_output to cuda, {input_.dtype}->fp16'
        )
        return input_.to(target_device).half()

    @staticmethod
    def backward(ctx, grad_output):
        args = get_args()
        return grad_output.to(torch.device('cpu:0')).float()


def copy_to_rank0(input_):
    return _GatherActToRank0.apply(input_)


def fetch_from_rank0(input_):
    return _BcastActFromRank0.apply(input_)


class CpuBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(CpuBertEmbeddings, self).__init__()
        self.bert_embedding = BertEmbeddingsWithoutLN(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0)

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

        input_ids_parallel = copy_to_rank0(input_ids)

        output_parallel = self.bert_embedding(input_ids_parallel,
                                              token_type_ids, position_ids)

        embeddings = fetch_from_rank0(output_parallel)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
