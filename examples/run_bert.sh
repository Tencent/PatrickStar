cd $(dirname $0)

export GPU_NUM=${GPU_NUM:-1}
# Chunk Size in MB
export CS=${CS:-64}
# Batch Size
export BS=${BS:-16}
# Embedding on CPU
export CPU_EBD=${CPU_EBD:-1}
# Release remote chunks after init
export RELEASE_AFTER_INIT=${RELEASE_AFTER_INIT:-0}
export MODEL_NAME=${MODEL_NAME:-"GPT2small"}
# distributed plan patrickstar or torch
export DIST_PLAN=${DIST_PLAN:-"patrickstar"}
# check results of patrickstar and torch, which disable
# DIST_PLAN setting
export RES_CHECK=${RES_CHECK:-0}
# offload activation checkpoints to CPU
export ACT_OFFLOAD=${ACT_OFFLOAD:-0}
# activation rematerization, aka. gradient checkpointing
export CKP=${CKP:-1}
# no retry after failed, used for torch 1.9.0
export NO_RETRY=${NO_RETRY:-0}
export SKIP_LOG_EXSIT=${SKIP_LOG_EXSIT:-0}
export AW=${AW:-0}
export MEM_PROF=${MEM_PROF:-0}

if [[ ${MEM_PROF} == 1 ]];  then
MEM_PROF_FLAG="--with_mem_profiler"
else
export MEM_PROF_FLAG=""
fi


if [[ ${ACT_OFFLOAD} == 1 ]];  then
ACT_OFFLOAD_FLAG="--with_activation_offload"
else
export ACT_OFFLOAD_FLAG=""
fi

if [[ ${RES_CHECK} == 1 ]];  then
RES_CHECK_FLAG="--res_check"
else
export RES_CHECK_FLAG=""
fi


if [[ ${CPU_EBD} == 1 ]];  then
export CPU_EBD_FLAG="--use_cpu_embedding"
else
export CPU_EBD_FLAG=""
fi

if [[ ${RELEASE_AFTER_INIT} == 1 ]];  then
export RELEASE_AFTER_INIT_FLAG="--release_after_init"
else
export RELEASE_AFTER_INIT_FLAG=""
fi

export LIGHTSEQ=0
if [[ ${LIGHTSEQ} == 1 ]]; then
export LIGHTSEQ_FLAG="--with_lightseq"
else
export LIGHTSEQ_FLAG=""
fi


if [[ ${CKP} == 1 ]]; then
    export CKP_FLAG="--use_ckp"
else
    export CKP_FLAG=""
fi

let CHUNK_SIZE=${CS}*1024*1024
export HYBRID_ADAM_FLAG="--use_hybrid_adam"

LOG_DIR="./logs_${MODEL_NAME}"
mkdir -p ${LOG_DIR}

LOG_FILE="log.${MODEL_NAME}_gpu_${GPU_NUM}_cs_${CS}_bs_${BS}_cpueb_${CPU_EBD}_lightseq_${LIGHTSEQ}_offload_${ACT_OFFLOAD}_AW_${AW}"

is_run_flag=`python ./benchmark/is_run_this_file.py --path "${LOG_DIR}" --file "${LOG_FILE}"`
echo is_run_flag $is_run_flag
if [[ ${is_run_flag} == "0" && ${SKIP_LOG_EXSIT} == 1 ]];
then
echo "it has been logged"
exit
fi
echo "runing ${LOG_DIR} ${LOG_FILE}"

if [[ ${NO_RETRY} == "1" ]];
then
NO_RETRY_FLAG="--max_restarts=0"
fi


if [[ ${AW} == 1 ]];
then
AW_FLAG="--always_warmup"
fi


python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
    pretrain_bert_demo.py \
    --use_fp16 \
    ${RES_CHECK_FLAG} \
    ${NO_RETRY_FLAG} \
    ${CKP_FLAG} \
    --dist_plan=${DIST_PLAN} \
    --batch_size=${BS} \
    --model_name=${MODEL_NAME} \
    --batch_size=${BS} \
    ${CPU_EBD_FLAG} \
    ${HYBRID_ADAM_FLAG} \
    ${RELEASE_AFTER_INIT_FLAG} \
    --default_chunk_size=${CHUNK_SIZE} \
    ${LIGHTSEQ_FLAG} \
    ${ACT_OFFLOAD_FLAG} \
    ${AW_FLAG} \
    ${MEM_PROF_FLAG} \
    2>&1 | tee ${LOG_DIR}/${LOG_FILE}
