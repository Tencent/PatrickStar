cd $(dirname $0)

export GPU_NUM=${GPU_NUM:-1}
# Chunk Size in MB
export CS=${CS:-64}
# Batch Size
export BS=${BS:-16}
# Release remote chunks after init
export RELEASE_AFTER_INIT=${RELEASE_AFTER_INIT:-0}
export MODEL_NAME=${MODEL_NAME:-"GPT2small"}
export MODEL_TYPE=${MODEL_TYPE:-"BERT"}
# distributed plan patrickstar or torch
export DIST_PLAN=${DIST_PLAN:-"patrickstar"}
# check results of patrickstar and torch, which disable
# DIST_PLAN setting
export RES_CHECK=${RES_CHECK:-0}
# activation rematerization, aka. gradient checkpointing
export CKP=${CKP:-1}
export FP16=${FP16:-1}
export SKIP_LOG_EXSIT=${SKIP_LOG_EXSIT:-0}
# asyn memory monitor for mem sampler
export AMM=${AMM:-1}
# mem caching comm
export CACHE=${CACHE:-1}

export LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE:-1}
export CS_SEARCH=${CS_SEARCH:-0}

export NNODES=${NNODES:-1}
export NODE_RANK=${NODE_RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
export MASTER_PORT=${MASTER_PORT:-"12345"}
export SUFFIX=${SUFFIX:-""}

if [[ ${AMM} == 1 ]];  then
AMM_FLAG="--with_async_mem_monitor"
else
export AMM_FLAG=""
fi

if [[ ${RES_CHECK} == 1 ]];  then
RES_CHECK_FLAG="--res_check"
else
export RES_CHECK_FLAG=""
fi

if [[ ${RELEASE_AFTER_INIT} == 1 ]];  then
export RELEASE_AFTER_INIT_FLAG="--release_after_init"
else
export RELEASE_AFTER_INIT_FLAG=""
fi

if [[ ${CKP} == 1 ]]; then
    export CKP_FLAG="--use_ckp"
else
    export CKP_FLAG=""
fi

if [[ ${FP16} == 1 ]]; then
    export FP16_FLAG="--use_fp16"
else
    export FP16_FLAG=""
fi

let CHUNK_SIZE=${CS}*1024*1024



LOG_DIR="./logs_${MODEL_NAME}"
mkdir -p ${LOG_DIR}

GIT_VER=`git rev-parse --short=5 HEAD`
LOG_FILE="log.${MODEL_NAME}_type_${MODEL_TYPE}_gpu_${GPU_NUM}_cs_${CS}_bs_${BS}_AMM_${AMM}_${GIT_VER}_node_${NNODES}_${SUFFIX}"

is_run_flag=`python ./benchmark/is_run_this_file.py --path "${LOG_DIR}" --file "${LOG_FILE}"`
echo is_run_flag $is_run_flag
if [[ ${is_run_flag} == "0" && ${SKIP_LOG_EXSIT} == 1 ]];
then
echo "it has been logged"
exit
fi
echo "runing ${LOG_DIR} ${LOG_FILE}"


wc=`cat /proc/cpuinfo | grep "processor"| wc -l`
let TNUM=wc/${GPU_NUM}
echo "CPU core number " $wc "THREAD NUM " ${TNUM}

cmd_opts="
    ${RES_CHECK_FLAG} \
    ${CKP_FLAG} \
    ${FP16_FLAG} \
    --dist_plan=${DIST_PLAN} \
    --batch_size=${BS} \
    --model_name=${MODEL_NAME} \
    --model_type=${MODEL_TYPE} \
    --batch_size=${BS} \
    ${RELEASE_AFTER_INIT_FLAG} \
    ${AMM_FLAG} \
"

if [[ ${CS_SEARCH} == 1 ]];  then
mkdir -p ./search_res
SLOG_FILE="./search_res/slog_file.${MODEL_NAME}_bs_${BS}_AMM_${AMM}_${GIT_VER}"
rm -rf ${SLOG_FILE}

for((i=312;i>=64;i-=32));
do
let CUR_CHUNK_SIZE=${i}*1024*1024
echo "searching CHUNK_SIZE ${i} M elem"

python -m torch.distributed.launch --nproc_per_node=1 \
    eval_chunk_size.py \
    --chunk_size=${CUR_CHUNK_SIZE} \
    --slog_file=${SLOG_FILE} \
    ${cmd_opts}
done
else
env OMP_NUM_THREADS=${TNUM} timeout -s SIGKILL 30m python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
--nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    pretrain_demo.py \
    --chunk_size=${CHUNK_SIZE} \
    ${cmd_opts} \
    2>&1 | tee ${LOG_DIR}/${LOG_FILE}
fi
