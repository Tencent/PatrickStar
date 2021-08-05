export GPU_NUM=${GPU_NUM:-4}
export MODEL_NAME="Bertlarge"
export CS=32
export BS=8

export margin_use_ratio=${margin_use_ratio:-0.8}
# if warmup fails, lower the ratio
export warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio:-0.2}
export overall_gpu_mem_ratio=${overall_gpu_mem_ratio:-0.8}

# Check result correctness
# RES_CHECK_FLAG="--res_check"
# Use a single GPU card to simulate multiple-GPU training.
# FAKE_DIST="--use_fake_dist"

export PYTHONPATH=../:${PYTHONPATH}

export HYBRID_ADAM_FLAG="--use_hybrid_adam"
export USE_DS_ADAM="--use_deepspeed_cpu_adam"
export CPU_EMBED="--use_cpu_embedding"
export CPU_EMBED_FP32="--cpu_embedding_fp32"

# for MODEL_NAME in "GPT3_8B" "GPT3_10B" "GPT3_12B" "GPT3_13B"
for MODEL_NAME in "GPT3_6B"
do
for BS in 32 16 8
do
for CS in 128 64 48 32
do
for CPU_EBD in 1 0
do
for AW in 0 1
do
let CHUNK_SIZE=${CS}*1024*1024

if [[ ${CPU_EBD} == 1 ]];  then
export CPU_EMBED="--use_cpu_embedding"
export CPU_EMBED_FP32="--cpu_embedding_fp32"
else
export CPU_EMBED=""
export CPU_EMBED_FP32=""
fi


if [[ ${AW} == 1 ]];  then
export always_warmup="--always_warmup"
else
export always_warmup=""
fi

export GPU_BOOST_ADAM=1

if [[ ${GPU_BOOST_ADAM} == 1 ]]; then
export use_gpu_fp32_convert_for_adam="--use_gpu_fp32_convert_for_adam"
else
export use_gpu_fp32_convert_for_adam=""
fi

echo "${CS} ${BS} ${MODEL_NAME}"
python ../patrickstar/launcher/runner.py --num_nodes 1 \
                             --num_gpus ${GPU_NUM} \
                             test_bert.py ${RES_CHECK_FLAG} \
                             --use_ckp \
                             --use_fp16 \
                             --use_ps \
                             ${use_gpu_fp32_convert_for_adam} \
                             --batch_size=${BS} \
                             --margin_use_ratio=${margin_use_ratio} \
                             --warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio} \
                             --model_name=${MODEL_NAME} \
                             ${FAKE_DIST} \
                             ${USE_DS_ADAM} \
                             ${CPU_EMBED} \
                             ${CPU_EMBED_FP32} \
                             ${HYBRID_ADAM_FLAG} \
                             ${always_warmup} \
                             --default_chunk_size=${CHUNK_SIZE} \
                             2>&1 | tee yard_logs_v2/log.${MODEL_NAME}_bs_${BS}_cs_${CS}_gpu_${GPU_NUM}_cpueb_${CPU_EBD}_aw_${AW}
done
done
done
done
done
