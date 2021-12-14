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

import transformers
from transformers import BertConfig
from packaging import version
import optimizations.global_opt_flags as global_opt_flags


def model_config(model_name):
    """
    generate the model config according to the model name.
    """
    if model_name == "Bert":
        # 0.11B
        HIDDEN_DIM = 768
        SEQ_LEN = 512
        NUM_LAYER = 6
        NUM_HEAD = 12
    elif model_name == "Bertlarge":
        # 0.35B
        HIDDEN_DIM = 1024
        SEQ_LEN = 512
        NUM_LAYER = 24
        NUM_HEAD = 16
    elif model_name == "GPT2small":
        # 0.7B
        HIDDEN_DIM = 1536
        SEQ_LEN = 128
        NUM_LAYER = 24
        NUM_HEAD = 16
    elif model_name == "GPT2_1B":
        # 0.9B
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 20
        NUM_HEAD = 16
    elif model_name == "megatron_1.3B":
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 24
        NUM_HEAD = 32
    elif model_name == "GPT2_2B":
        # zero-offload
        HIDDEN_DIM = 2048
        SEQ_LEN = 1024
        NUM_LAYER = 40
        NUM_HEAD = 16
    elif model_name == "megatron_3.9B":
        # Table 4 in Megatron Paper
        HIDDEN_DIM = 2560
        SEQ_LEN = 1024
        NUM_LAYER = 24
        NUM_HEAD = 40
    elif model_name == "GPT2_4B":
        HIDDEN_DIM = 2304  # 2048
        SEQ_LEN = 1024
        NUM_LAYER = 64
        NUM_HEAD = 16
    elif model_name == "GPT3_6B":
        # 6.7B model
        HIDDEN_DIM = 3072
        SEQ_LEN = 1024
        NUM_LAYER = 53
        NUM_HEAD = 16
    elif model_name == "GPT3_8B":
        # 6.7B model
        HIDDEN_DIM = 3072
        SEQ_LEN = 1024
        NUM_LAYER = 72
        NUM_HEAD = 16
    elif model_name == "GPT3_10B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 50
        NUM_HEAD = 16
    elif model_name == "GPT3_11B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 55
        NUM_HEAD = 16
    elif model_name == "GPT3_12B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 60
        NUM_HEAD = 16
    elif model_name == "GPT3_13B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 65
        NUM_HEAD = 16
    elif model_name == "GPT3_15B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 78
        NUM_HEAD = 16
    elif model_name == "GPT3_18B":
        HIDDEN_DIM = 4096
        SEQ_LEN = 1024
        NUM_LAYER = 90
        NUM_HEAD = 16
    # The following configs comes from paper
    # Efficient Large-Scale Language Model Training on GPU Clusters
    # NV model is wider in hidden-size
    elif model_name == "GPT_NV_18B":
        HIDDEN_DIM = 6144
        SEQ_LEN = 1024
        NUM_LAYER = 40
        NUM_HEAD = 16
    elif model_name == "GPT_NV_39B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 48
        NUM_HEAD = 16
    elif model_name == "GPT_NV_76B":
        HIDDEN_DIM = 10240
        SEQ_LEN = 1024
        NUM_LAYER = 60
        NUM_HEAD = 16
    # The following configs comes from Deep-Offload
    # http://pasalabs.org/papers/2021/ATC21_zero-offload.pdf
    elif model_name == "GPT_DS_20B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 25
        NUM_HEAD = 16
    elif model_name == "GPT_DS_40B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 50
        NUM_HEAD = 16
    elif model_name == "GPT_DS_50B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 62
        NUM_HEAD = 16
    elif model_name == "GPT_DS_60B":
        HIDDEN_DIM = 8192
        SEQ_LEN = 1024
        NUM_LAYER = 75
        NUM_HEAD = 16
    elif model_name == "GPT_DS_68B":
        HIDDEN_DIM = 9216
        SEQ_LEN = 1024
        NUM_LAYER = 66
        NUM_HEAD = 16
    # OpenAI GPT3
    elif model_name == "GPT_175B":
        HIDDEN_DIM = 12288
        SEQ_LEN = 1024
        NUM_LAYER = 96
        NUM_HEAD = 96
    elif model_name == "GPT_310B":
        HIDDEN_DIM = 16384
        SEQ_LEN = 1024
        NUM_LAYER = 128
        NUM_HEAD = 128
    elif model_name == "GPT_454B":
        HIDDEN_DIM = 20480
        SEQ_LEN = 1024
        NUM_LAYER = 90  # 105 for 530B
        NUM_HEAD = 128
    else:
        raise RuntimeError(f"The model name {model_name} is not valid!")
    assert HIDDEN_DIM % NUM_HEAD == 0
    return (HIDDEN_DIM, SEQ_LEN, NUM_LAYER, NUM_HEAD)


def print_model_config(args, hidden_dim, sequence_len, num_layer, num_head):
    if args.rank == 0:
        config_dict = {
            "hidden_dim": hidden_dim,
            "sequence_len": sequence_len,
            "num_layer": num_layer,
            "num_head": num_head,
        }
        print("------------------ model config ------------------", flush=True)
        str_list = []
        for key, value in config_dict.items():
            dots = "." * (32 - len(key))
            str_list.append("  {} {} {}".format(key, dots, value))
        for arg in sorted(str_list, key=lambda x: x.lower()):
            print(arg, flush=True)
        print("-------------- end of model config --------------", flush=True)


def build_transformer_model(args):
    """
    Build a transformer-based model based on transformer bert.
    return a function able to build the model.
    """
    if args.with_tiling_linear or args.with_activation_offload:
        if args.with_tiling_linear:
            global_opt_flags.USE_TILE = True
        else:
            global_opt_flags.USE_TILE = False
        if args.with_activation_offload:
            global_opt_flags.USE_ACT_OFFLOAD = True
        else:
            global_opt_flags.USE_ACT_OFFLOAD = False
        from optimizations.ps_tile_modeling_bert import BertForSequenceClassification
    else:
        from transformers import BertForSequenceClassification

    hidden_dim, sequence_length, num_layer, num_head = model_config(args.model_name)

    bert_config = BertConfig(
        gradient_checkpointing=args.use_ckp,
        hidden_size=hidden_dim,
        intermediate_size=hidden_dim * 4,
        num_attention_heads=num_head,
        max_position_embeddings=sequence_length,
        num_hidden_layers=num_layer,
    )

    def model_func():
        model = BertForSequenceClassification(bert_config)
        if args.use_ckp and version.parse(transformers.__version__) >= version.parse(
            "4.11.0"
        ):
            model.gradient_checkpointing_enable()
        return model

    return model_func, sequence_length
