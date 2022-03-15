#!/bin/bash
set -e
cd "$(dirname "$0")"

export GPU_NUM=${GPU_NUM:-1}
export MODEL_NAME=${MODEL_NAME:-"GPT2small"}
export MODEL_TYPE=${MODEL_TYPE:-"BERT"}
# Chunk Size in MB
export CS=${CS:-64}
# Batch Size
export BS=${BS:-16}
# Release remote chunks after init
export RELEASE_AFTER_INIT=${RELEASE_AFTER_INIT:-0}
# check results of PatrickStar and torch, which disable
export RES_CHECK=${RES_CHECK:-0}

export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"12345"}

if [[ ${RES_CHECK} == 1 ]];  then
  RES_CHECK_FLAG="--res_check"
else
  RES_CHECK_FLAG=""
fi

if [[ ${RELEASE_AFTER_INIT} == 1 ]];  then
  RELEASE_AFTER_INIT_FLAG="--release_after_init"
else
  RELEASE_AFTER_INIT_FLAG=""
fi

(( CHUNK_SIZE = CS * 1024 * 1024 )) || true

LOG_DIR="./logs_${MODEL_NAME}"
mkdir -p "${LOG_DIR}"

GIT_VER=$(git rev-parse --short=5 HEAD)
LOG_FILE="log.${MODEL_NAME}_type_${MODEL_TYPE}_gpu_${GPU_NUM}_cs_${CS}_bs_${BS}_AMM_${AMM}_${GIT_VER}_node_${NNODES}"



(( TNUM=$(nproc)/GPU_NUM )) || true
echo "CPU core number " $(nproc) "THREAD NUM " "${TNUM}"

env OMP_NUM_THREADS="${TNUM}" python -m torch.distributed.launch \
    --nproc_per_node="${GPU_NUM}" --nnodes="${NNODES}" --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    pretrain_demo.py \
    --chunk_size="${CHUNK_SIZE}" \
    --batch_size="${BS}" \
    --model_name="${MODEL_NAME}" \
    --model_type="${MODEL_TYPE}" \
    "${RES_CHECK_FLAG}" \
    "${RELEASE_AFTER_INIT_FLAG}" \
    2>&1 | tee "${LOG_DIR}/${LOG_FILE}"
