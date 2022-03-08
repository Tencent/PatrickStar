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

from transformers import BertLayer, BertForSequenceClassification
from transformers.models.bert.modeling_bert import BertAttention

from patrickstar.core import torch_scope
from patrickstar.utils import logger, get_world_size

try:
    import fmoe
except ImportError:
    logger.error("Please install FastMoE to use MoE with PatrickStar.")


def __init__(self, config):
    super(BertLayer, self).__init__()
    self.chunk_size_feed_forward = config.chunk_size_feed_forward
    self.seq_len_dim = 1
    self.attention = BertAttention(config)
    self.is_decoder = config.is_decoder
    self.add_cross_attention = config.add_cross_attention
    if self.add_cross_attention:
        assert (
            self.is_decoder
        ), f"{self} should be used as a decoder model if cross attention is added"
        self.crossattention = BertAttention(config)
    # The MoE modules are mainly of model parallel, we need to use `torch_scope`
    # to separate it from the other chunk based data parallel modules.
    # Also, MoE modules will take cart of its own communication, that's why
    # we need to disable allreduce in the torch scope.
    with torch_scope(do_allreduce=False):
        self.output = fmoe.FMoETransformerMLP(
            num_expert=2,
            world_size=get_world_size(),
            d_model=config.hidden_size,
            d_hidden=config.intermediate_size,
            gate=fmoe.gates.NaiveGate,
        )


def feed_forward_chunk(self, attention_output):
    layer_output = self.output(attention_output)
    return layer_output


def build_moe_bert():
    # Normally you should write your own Model and create the MoE parts
    # in it. Here we directly substitute the origin huggingface Bert model
    # for simplicity.
    BertLayer.__init__ = __init__
    BertLayer.feed_forward_chunk = feed_forward_chunk
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    return model
