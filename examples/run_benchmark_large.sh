export GPU_NUM=${GPU_NUM:-1}
export CS=${CS:-64}
export BS=${BS:-16}
export CPU_EBD=${CPU_EBD:-1}
export RELEASE_AFTER_INIT=${RELEASE_AFTER_INIT:-0}
export MODEL_NAME=${MODEL_NAME:-"GPT2small"}
export DIST_PLAN=${DIST_PLAN:-"patrickstar"}
export RES_CHECK=${RES_CHECK:-0}
export ACT_OFFLOAD=${ACT_OFFLOAD:-0}

export margin_use_ratio=${margin_use_ratio:-0.8}
# if warmup fails, lower the ratio
export warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio:-0.1}
export overall_gpu_mem_ratio=${overall_gpu_mem_ratio:-0.8}

if [[ ${RES_CHECK} == 1 ]];  then
# Check result correctness
RES_CHECK_FLAG="--res_check"
else
export RES_CHECK_FLAG=""
fi


export HYBRID_ADAM_FLAG="--use_hybrid_adam"

if [[ ${RELEASE_AFTER_INIT} == 1 ]];  then
export RELEASE="--release_after_init"
else
export RELEASE=""
fi

export LIGHTSEQ=0
if [[ ${LIGHTSEQ} == 1 ]]; then
export lightseq_flag="--with_lightseq"
else
export lightseq_flag=""
fi

mkdir -p ./logs

for MODEL_NAME in "GPT3_8B" "GPT3_10B" "GPT3_12B" "GPT3_11B"
do
for BS in 32 16 8
do
for CS in 64 128
do
let CHUNK_SIZE=${CS}*1024*1024
for CPU_EBD in 1 0
do
if [[ ${CPU_EBD} == 1 ]];  then
export CPU_EMBED="--use_cpu_embedding"
else
export CPU_EMBED=""
fi

for AW in 0 1
do
if [[ ${AW} == 1 ]];  then
export always_warmup="--always_warmup"
else
export always_warmup=""
fi


for ACT_OFFLOAD in 0 1
do
if [[ ${ACT_OFFLOAD} == 1 ]];  then
# Check result correctness
ACT_OFFLOAD_FLAG="--with_activation_offload"
else
export ACT_OFFLOAD_FLAG=""
fi

echo "benchmarking ${CS} ${BS} ${MODEL_NAME} ${CPU_EBD} ${AW} ${ACT_OFFLOAD}"
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
                           --max_restarts=0 \
                             pretrain_bert_demo.py ${RES_CHECK_FLAG} \
                             --use_ckp \
                             --use_fp16 \
                             --dist_plan=${DIST_PLAN} \
                             --batch_size=${BS} \
                             --model_name=${MODEL_NAME} \
                             --overall_gpu_mem_ratio=${overall_gpu_mem_ratio} \
                             --batch_size=${BS} \
                             --margin_use_ratio=${margin_use_ratio} \
                             --warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio} \
                             ${CPU_EMBED} \
                             ${HYBRID_ADAM_FLAG} \
                             ${RELEASE} \
                             --default_chunk_size=${CHUNK_SIZE} \
                             ${lightseq_flag} \
                             ${ACT_OFFLOAD_FLAG} \
           ${always_warmup} \
                             2>&1 | tee ./logs/log.${MODEL_NAME}_gpu_${GPU_NUM}_cs_${CS}_bs_${BS}_cpueb_${CPU_EBD}_margin_${margin_use_ratio}_warmup_${warmup_gpu_chunk_mem_ratio}_gpu_${overall_gpu_mem_ratio}_lightseq_${LIGHTSEQ}_offload_${ACT_OFFLOAD}_aw_${AW}

done
done
done
done
done
done
