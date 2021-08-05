
export GPU_NUM=${GPU_NUM:-1}
export CS=${CS:-64}
export BS=${BS:-16}
export CPU_EBD=${CPU_EBD:-1}
export MODEL_NAME=${MODEL_NAME:-"Bertlarge"}

export margin_use_ratio=${margin_use_ratio:-0.8}
# if warmup fails, lower the ratio
export warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio:-0.2}
export overall_gpu_mem_ratio=${overall_gpu_mem_ratio:-0.8}

# Check result correctness
# RES_CHECK_FLAG="--res_check"
# Use a single GPU card to simulate multiple-GPU training.
# FAKE_DIST="--use_fake_dist"

let CHUNK_SIZE=${CS}*1024*1024
export PYTHONPATH=../:${PYTHONPATH}

export HYBRID_ADAM_FLAG="--use_hybrid_adam"
export USE_DS_ADAM="--use_deepspeed_cpu_adam"

if [[ ${CPU_EBD} == 1 ]];  then
export CPU_EMBED="--use_cpu_embedding"
export CPU_EMBED_FP32="--cpu_embedding_fp32"
else
export CPU_EMBED=""
export CPU_EMBED_FP32=""
fi
# export MODEL_NAME="GPTsmall"
# export BS=4
python ../patrickstar/launcher/runner.py --num_nodes 1 \
                             --num_gpus ${GPU_NUM} \
                             test_bert.py ${RES_CHECK_FLAG} \
                             --use_ckp \
                             --use_fp16 \
                             --use_ps \
                             --batch_size=${BS} \
                             --model_name=${MODEL_NAME} \
                             --overall_gpu_mem_ratio=${overall_gpu_mem_ratio} \
                             --batch_size=${BS} \
                             --margin_use_ratio=${margin_use_ratio} \
                             --warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio} \
                             ${FAKE_DIST} \
                             ${USE_DS_ADAM} \
                             ${CPU_EMBED} \
                             ${CPU_EMBED_FP32} \
                             ${HYBRID_ADAM_FLAG} \
                             --default_chunk_size=${CHUNK_SIZE} \
                             2>&1 | tee ./logs/log.${MODEL_NAME}_gpu_${GPU_NUM}_cs_${CS}_bs_${BS}_cpueb_${CPU_EBD}_margin_${margin_use_ratio}_warmup_${warmup_gpu_chunk_mem_ratio}_gpu_${overall_gpu_mem_ratio}
