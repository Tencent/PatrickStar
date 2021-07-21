#! /bin/bash

# Change for multinode config
MP_SIZE=1
NUM_GPUS_PER_WORKER=1
NUM_WORKERS=1

BASE_DATA_PATH=/data/Megatron-LM/data
DATA_PATH=${BASE_DATA_PATH}/indexed_datasets/megatron
VOCAB_PATH=${BASE_DATA_PATH}/gpt2-vocab.json
MERGE_PATH=${BASE_DATA_PATH}/gpt2-merges.txt
CHECKPOINT_PATH=checkpoints/gpt2_345m_ds

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
if [[ -z $1 ]]; then
       config_json="$script_dir/ds_zero_stage_3_config.json"

       # offloads to NVMe
       #config_json="$script_dir/ds_zero_stage_infinity_config.json"
else
       config_json=$script_dir/`basename $1`
fi

#ZeRO Configs
stage=3
reduce_scatter=true
contigious_gradients=true
rbs=50000000
agbs=5000000000

#Activation Checkpointing and Contigious Memory
chkp_layers=1
PA=true
PA_CPU=true
CC=true
SYNCHRONIZE=true
PROFILE=false

# TiledLinear splits, 0 is disable
TILED_LINEAR="false"
TILE_DIM=1


# Megatron Model Parallelism
LOGDIR="tboard-zero3/stage${stage}g"

 deepspeed_options=" \
                --deepspeed \
                --deepspeed_config ${config_json} \
                --zero-stage ${stage} \
                --zero-reduce-bucket-size ${rbs} \
                --zero-allgather-bucket-size ${agbs}
            "

if [ "${contigious_gradients}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-contigious-gradients"
fi

if [ "${reduce_scatter}" = "true" ]; then
deepspeed_options="${deepspeed_options} \
                --zero-reduce-scatter"
fi

chkp_opt=" \
--deepspeed-activation-checkpointing \
--checkpoint-num-layers ${chkp_layers}"

if [ "${PA}" = "true" ]; then
chkp_opt="${chkp_opt} --partition-activations"
fi

if [ "${PA_CPU}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --checkpoint-in-cpu"
fi

if [ "${SYNCHRONIZE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --synchronize-each-layer"
fi

if [ "${CC}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --contigious-checkpointing"
fi

if [ "${PROFILE}" = "true" ]; then
chkp_opt="${chkp_opt} \
        --profile-backward"
fi

if [ "${TILED_LINEAR}" = "true" ]; then
tile_opt="${tile_opt} \
        --memory-centric-tiled-linear \
        --tile-factor=${TILE_DIM}"
fi


full_options="${deepspeed_options}"

#  ${chkp_opt} ${tile_opt}
export MODEL_NAME="Bert"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} test_bert.py --use_ckp --use_ds ${@:2} ${full_options} --model_name=${MODEL_NAME}"
echo ${run_cmd}
eval ${run_cmd}

set +x
