
export GPU_NUM=${GPU_NUM:-1}
export CS=${CS:-64}
export BS=${BS:-16}
export CPU_EBD=${CPU_EBD:-1}
export RELEASE_AFTER_INIT=${RELEASE_AFTER_INIT:-0}
export MODEL_NAME=${MODEL_NAME:-"GPT2small"}
export RES_CHECK=${RES_CHECK:-1}

export margin_use_ratio=${margin_use_ratio:-0.8}
# if warmup fails, lower the ratio
export warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio:-0.2}
export overall_gpu_mem_ratio=${overall_gpu_mem_ratio:-0.8}

if [[ ${RES_CHECK} == 1 ]];  then
# Check result correctness
export RES_CHECK_FLAG="--res_check"
else
export RES_CHECK_FLAG=""
fi
# Use a single GPU card to simulate multiple-GPU training.
# FAKE_DIST="--use_fake_dist"

let CHUNK_SIZE=${CS}*1024*1024

export HYBRID_ADAM_FLAG="--use_hybrid_adam"

if [[ ${CPU_EBD} == 1 ]];  then
export CPU_EMBED="--use_cpu_embedding"
else
export CPU_EMBED=""
fi

if [[ ${RELEASE_AFTER_INIT} == 1 ]];  then
export RELEASE="--release_after_init"
else
export RELEASE=""
fi

mkdir -p ./logs
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
                             pretrain_bert_demo.py ${RES_CHECK_FLAG} \
                             --use_ckp \
                             --use_fp16 \
                             --dist_plan="patrickstar" \
                             --batch_size=${BS} \
                             --model_name=${MODEL_NAME} \
                             --overall_gpu_mem_ratio=${overall_gpu_mem_ratio} \
                             --batch_size=${BS} \
                             --margin_use_ratio=${margin_use_ratio} \
                             --warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio} \
                             ${FAKE_DIST} \
                             ${USE_DS_ADAM} \
                             ${CPU_EMBED} \
                             ${HYBRID_ADAM_FLAG} \
                             ${RELEASE} \
                             --default_chunk_size=${CHUNK_SIZE} \
                             2>&1 | tee ./logs/log.${MODEL_NAME}_gpu_${GPU_NUM}_cs_${CS}_bs_${BS}_cpueb_${CPU_EBD}_margin_${margin_use_ratio}_warmup_${warmup_gpu_chunk_mem_ratio}_gpu_${overall_gpu_mem_ratio}
