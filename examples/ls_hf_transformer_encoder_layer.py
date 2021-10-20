# BSD 3-Clause License
#
# Copyright (C) 2021 THL A29 Limited, a Tencent company.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
#  * Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
#  * Neither the name of the psutil authors nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

try:
    from lightseq.training.ops.pytorch.transformer_encoder_layer import (
        LSTransformerEncoderLayer,
    )
except ImportError:
    raise RuntimeError("pip install lightseq first!")


class LSHFTransformerEncoderLayer(LSTransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super(LSHFTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, hidden_states, encoder_padding_mask, *args, **kwargs):
        encoder_padding_mask /= -10000.0
        encoder_padding_mask = encoder_padding_mask.squeeze()
        output = super().forward(hidden_states, encoder_padding_mask)
        return (output, None, None, None)


def gen_bert_config(training_args, config):
    bert_config = LSTransformerEncoderLayer.get_config(
        max_batch_tokens=4096,
        max_seq_len=config.max_position_embeddings,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        nhead=config.num_attention_heads,
        attn_prob_dropout_ratio=config.attention_probs_dropout_prob,
        activation_dropout_ratio=config.hidden_dropout_prob,
        hidden_dropout_ratio=config.hidden_dropout_prob,
        pre_layer_norm=False,
        fp16=training_args.use_fp16,
        local_rank=training_args.local_rank,
        activation_fn="gelu",
    )
    return bert_config


def get_hf_bert_enc_layer_params(layer):
    init_ws = []
    init_bs = []

    init_ws.append(layer.attention.self.query.weight.detach().clone())
    init_bs.append(layer.attention.self.query.bias.detach().clone())
    init_ws.append(layer.attention.self.key.weight.detach().clone())
    init_bs.append(layer.attention.self.key.bias.detach().clone())
    init_ws.append(layer.attention.self.value.weight.detach().clone())
    init_bs.append(layer.attention.self.value.bias.detach().clone())
    init_ws.append(layer.attention.output.dense.weight.detach().clone())
    init_bs.append(layer.attention.output.dense.bias.detach().clone())
    init_ws.append(layer.attention.output.LayerNorm.weight.detach().clone())
    init_bs.append(layer.attention.output.LayerNorm.bias.detach().clone())

    init_ws.append(layer.intermediate.dense.weight.detach().clone())
    init_bs.append(layer.intermediate.dense.bias.detach().clone())
    init_ws.append(layer.output.dense.weight.detach().clone())
    init_bs.append(layer.output.dense.bias.detach().clone())
    init_ws.append(layer.output.LayerNorm.weight.detach().clone())
    init_bs.append(layer.output.LayerNorm.bias.detach().clone())

    return init_ws, init_bs


def inject_ls_enc_layer(model, training_args, config):
    for i in range(config.num_hidden_layers):
        bert_config = gen_bert_config(training_args, config)
        init_ws, init_bs = get_hf_bert_enc_layer_params(model.bert.encoder.layer[i])
        model.bert.encoder.layer[i] = LSHFTransformerEncoderLayer(
            bert_config, init_ws, init_bs
        ).cuda()
