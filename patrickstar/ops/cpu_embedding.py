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
    if torch.distributed.get_world_size() == 1:
        return
    comm_group = get_embedding_comm_group()
    torch.distributed.send(input_, dst=tgt_rank, group=comm_group)
    return


def _recv_from_rank(buff, src_rank):
    if torch.distributed.get_world_size() == 1:
        return buff
    comm_group = get_embedding_comm_group()

    torch.distributed.recv(buff, src=src_rank, group=comm_group)
    return buff


def _bcast_from_rank(input_tensor, src_rank):
    """
    将src_rank进程上的input tensor切成N份，发送到其他rank的进程上
    input tensor在cpu，输出结果再gpu上
    src_rank: input_tensor是一个batch的activation
    其他rank: input_tensor是一个进程的activation
    """
    work_device = torch.device('cpu:0')
    input_tensor = input_tensor.cpu()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    comm_group = get_embedding_comm_group()

    if local_rank == src_rank:
        splitted_tensor = torch.split(input_tensor,
                                      input_tensor.shape[0] // world_size)
        scatter_list = []
        for i in range(world_size):
            scatter_list.append(splitted_tensor[i])
        torch.distributed.scatter(scatter_list[src_rank],
                                  scatter_list,
                                  src=src_rank,
                                  group=comm_group)
        return scatter_list[src_rank]
    else:
        torch.distributed.scatter(input_tensor, src=src_rank, group=comm_group)
        return input_tensor


def _gather_to_rank(input_tensor, tgt_rank):
    """
    所有rank不是tgt_rank的input_tensor gather到tgt_rank上
    这个操作调用Gloo的API，所以输入数据需要在cpu上
    """
    work_device = torch.device('cpu:0')
    input_tensor = input_tensor.cpu()
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    comm_group = get_embedding_comm_group()
    if local_rank == tgt_rank:
        gather_list = []
        for i in range(world_size):
            if i == tgt_rank:
                gather_list.append(input_tensor)
            else:
                gather_list.append(torch.zeros_like(input_tensor))
        torch.distributed.gather(input_tensor,
                                 gather_list,
                                 dst=tgt_rank,
                                 group=comm_group)
        res = torch.tensor([], dtype=input_tensor.dtype, device=work_device)
        for t in gather_list:
            res = torch.cat((res, t), 0)
        return res
    else:
        torch.distributed.gather(input_tensor, dst=tgt_rank, group=comm_group)
        return torch.tensor([], dtype=input_tensor.dtype, device=work_device)


class _GatherToRank0(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        """
        把input_都聚合到rank 0
        rank 0返回一个拼接后的tensor
        其它rank返回空的tensor
        """
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return input_.to(torch.device('cpu:0'))

        input_ = input_.to(torch.device('cpu:0'))
        return _gather_to_rank(input_, 0)

    @staticmethod
    def forward(ctx, input_):
        return _GatherToRank0.symbolic(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError({"Not Implemented"})
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        logger.info(
            'Entrying CPU Emedding BWD, copy grad_output to cuda, fp32->fp16')
        return grad_output.to(target_device)


class _BcastFromRank0(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_):
        """
        输入分布在cpu rank=0上，按照batch维度拆分成N份，分发给各个进程
        @args
            input_: cpu上的activation tensor，是聚合后的
        返回本进程的activations
        """
        target_device = torch.device(f'cuda:{torch.cuda.current_device()}')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return input_.to(target_device)

        input_ = input_.cpu()
        return _bcast_from_rank(input_, 0)

    @staticmethod
    def forward(ctx, input_):
        return _BcastFromRank0.symbolic(ctx, input_)

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output每个进程都有一份，需要聚合成一份放在rank0上
        """
        work_device = torch.device(f'cpu:0')
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        if world_size == 1:
            return grad_output.to(work_device).float()

        grad_output = grad_output.to(work_device)
        return _gather_to_rank(grad_output, 0)


def send_ids_to_rank0(input_):
    return _GatherToRank0.apply(input_)


def collect_act_from_rank0(input_):
    return _BcastFromRank0.apply(input_)


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
        world_size = torch.distributed.get_world_size()
        new_input_ids = send_ids_to_rank0(input_ids)

        if global_rank == 0:
            # run embedding on CPU, output_activation (fp32 on CPU)
            output_activation = self.bert_embedding(new_input_ids,
                                                    token_type_ids,
                                                    position_ids)
        else:
            output_activation = torch.zeros(
                new_input_ids.shape[0] // world_size,
                self.bert_embedding.hidden_size)
        embeddings = collect_act_from_rank0(output_activation)
        assert embeddings.dtype == torch.float, f"embedding outputs should be in float on CPU, now {embeddings.dtype}"

        if is_param_registered(self.LayerNorm.weight):
            tgt_type = self.LayerNorm.weight.ps_attr.data_type
            assert tgt_type == torch.half, f"{self.LayerNorm.weight.ps_attr.name}"
        else:
            tgt_type = self.LayerNorm.weight.dtype
        embeddings = self.LayerNorm(embeddings.to(tgt_type))
        embeddings = self.dropout(embeddings)
        return embeddings
