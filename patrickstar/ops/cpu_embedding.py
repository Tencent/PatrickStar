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
from patrickstar.utils import logger
from patrickstar.core.parameter import is_param_registered
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
        device = torch.device(f'cuda:{torch.cuda.current_device()}')

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


# Model parallel group that the current rank belongs to.
_EMBEDING_COMM_GROUP = None


def get_embedding_comm_group():
    global _EMBEDING_COMM_GROUP
    if _EMBEDING_COMM_GROUP is None:
        _EMBEDING_COMM_GROUP = torch.distributed.new_group(backend='gloo')
    assert _EMBEDING_COMM_GROUP is not None, \
        'embedding communication group is not initialized'
    return _EMBEDING_COMM_GROUP


def _send_to_rank(input_, tgt_rank):
    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size() == 1:
        return
    comm_group = get_embedding_comm_group()
    # All-reduce.
    torch.distributed.send(input_, dst=tgt_rank, group=comm_group)
    return


def _recv_from_rank(buff, src_rank):
    # Bypass the function if we are using only 1 GPU.
    if torch.distributed.get_world_size() == 1:
        return buff
    comm_group = get_embedding_comm_group()
    # All-reduce.
    print('buff', buff, buff.dtype)
    torch.distributed.recv(buff, src=src_rank, group=comm_group)
    return buff


class _SendTo(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return input_.to(torch.device('cpu:0'))

        input_ = input_.to(torch.device('cpu:0'))
        if rank == 0:
            for i in range(1, world_size):
                outputs_ = _recv_from_rank(
                    torch.zeros_like(input_, device=torch.device('cpu:0')), i)
                input_ = torch.cat((input_, outputs_), 0)
            return input_
        else:
            _send_to_rank(input_, 0)
            return input_

    @staticmethod
    def forward(ctx, input_):
        return _SendTo.symbolic(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError({"Not Implemented"})
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        logger.info(
            'Entrying CPU Emedding BWD, copy grad_output to cuda, fp32->fp16')
        return grad_output.to(target_device)


class _CollectFrom(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        """
        输入分布在cpu rank=0上，按照batch维度拆分成N份，分发给各个进程
        返回本进程的activations
        """
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return input_.to(target_device)

        input_ = input_.cpu()
        mini_batch_inputs_ = torch.split(input_, input_.shape[0] // world_size)
        if local_rank == 0:
            for i in range(1, world_size):
                _send_to_rank(mini_batch_inputs_[i], i)
            return mini_batch_inputs_[0].to(target_device)
        else:
            input_ = _recv_from_rank(
                torch.zeros_like(mini_batch_inputs_[local_rank]), 0)
            return input_.to(target_device)

    @staticmethod
    def forward(ctx, input_):
        return _CollectFrom.symbolic(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):
        target_device = torch.device(f'cpu:0')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return grad_output.to(torch.device('cpu:0')).float()

        grad_output = grad_output.to(target_device)
        if local_rank != 0:
            _send_to_rank(grad_output, 0)
            return grad_output
        else:
            for i in range(1, world_size):
                output_ = _recv_from_rank(torch.zeros_like(grad_output), i)
                torch.cat((grad_output, output_), 0)
            return grad_output


def send_ids_to_parallel_region(input_):
    return _SendTo.apply(input_)


def collect_act_from_parallel_region(input_):
    return _CollectFrom.apply(input_)


class CpuBertEmbeddings(nn.Module):
    def __init__(self, config):
        super(CpuBertEmbeddings, self).__init__()
        self.bert_embedding = BertEmbeddingsWithoutLN(config)
        self.LayerNorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0)
        global _EMBEDING_COMM_GROUP
        _EMBEDING_COMM_GROUP = torch.distributed.new_group(backend='gloo')

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

        global_rank = torch.distributed.get_rank()

        new_input_ids = send_ids_to_parallel_region(input_ids)

        if global_rank == 0:
            output_activation = self.bert_embedding(new_input_ids,
                                                    token_type_ids,
                                                    position_ids)

        embeddings = collect_act_from_parallel_region(output_activation)
        assert embeddings.dtype == torch.float, f"embedding outputs should be in float on CPU, now {embeddings.dtype}"

        if is_param_registered(self.LayerNorm.weight):
            tgt_type = self.LayerNorm.weight.ps_attr.data_type
            assert tgt_type == torch.half, f"{self.LayerNorm.weight.ps_attr.name}"
        else:
            tgt_type = self.LayerNorm.weight.dtype
        embeddings = self.LayerNorm(embeddings.to(tgt_type))
        embeddings = self.dropout(embeddings)
        return embeddings
