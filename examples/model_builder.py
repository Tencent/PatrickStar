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

from packaging import version

import transformers
from transformers import BertConfig, GPT2Config
from transformers import BertForSequenceClassification, GPT2ForSequenceClassification


def model_config(model_name):
    """
    generate the model config according to the model name.
    """
    if model_name == "Tiny":
        HIDDEN_DIM = 16
        SEQ_LEN = 8
        NUM_LAYER = 1
        NUM_HEAD = 4
    elif model_name == "Bert":
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
    elif model_name == "GPT_220B":
        HIDDEN_DIM = 12288
        SEQ_LEN = 1024
        NUM_LAYER = 120
        NUM_HEAD = 96
    elif model_name == "GPT_250B":
        HIDDEN_DIM = 12288
        SEQ_LEN = 1024
        NUM_LAYER = 137
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
    if args.model_type.upper() == "BERT":
        Model = BertForSequenceClassification
    elif args.model_type.upper() == "GPT":
        Model = GPT2ForSequenceClassification

    hidden_dim, sequence_length, num_layer, num_head = model_config(args.model_name)

    if args.model_type.upper() == "BERT":
        config = BertConfig(
            gradient_checkpointing=args.use_ckp,
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 4,
            num_attention_heads=num_head,
            max_position_embeddings=sequence_length,
            num_hidden_layers=num_layer,
        )
    elif args.model_type.upper() == "GPT":
        config = GPT2Config(
            gradient_checkpointing=args.use_ckp,
            hidden_size=hidden_dim,
            intermediate_size=hidden_dim * 4,
            num_attention_heads=num_head,
            max_position_embeddings=sequence_length,
            num_hidden_layers=num_layer,
        )
    else:
        raise RuntimeError(
            f"Unknown model_type {args.model_type}, possible values are 'BERT' and 'GPT'"
        )

    def model_func():
        model = Model(config)
        # Need to set pad_token_id for batch size > 1.
        if args.model_type.upper() == "GPT":
            model.config.pad_token_id = model.config.eos_token_id

        if args.use_ckp and version.parse(transformers.__version__) >= version.parse(
            "4.11.0"
        ):
            model.gradient_checkpointing_enable()
        return model

    return model_func, sequence_length
