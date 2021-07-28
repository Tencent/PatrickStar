export GPU_NUM=1
export MODEL_NAME="Bertlarge"
export CS=32
export BS=8

# Check result correctness
# RES_CHECK_FLAG="--res_check"
# Use a single GPU card to simulate multiple-GPU training.
# FAKE_DIST="--use_fake_dist"

export PYTHONPATH=../:${PYTHONPATH}

export HYBRID_ADAM_FLAG="--use_hybrid_adam"
export USE_DS_ADAM="--use_deepspeed_cpu_adam"
export CPU_EMBED="--use_cpu_embedding"
export CPU_EMBED_FP32="--cpu_embedding_fp32"

for MODEL_NAME in "GPT2small" #"GPT2_1B" "GPT2_2B" "GPT2_4B"
do
for BS in 32
do
for CS in 32
do
let CHUNK_SIZE=${CS}*1024*1024

echo "${CS} ${BS} ${MODEL_NAME}"
python ../patrickstar/launcher/runner.py --num_nodes 1 \
                             --num_gpus ${GPU_NUM} \
                             test_bert.py ${RES_CHECK_FLAG} \
                             --use_ckp \
                             --use_fp16 \
                             --use_ps \
                             --batch_size=${BS} \
                             --model_name=${MODEL_NAME} \
                             ${FAKE_DIST} \
                             ${USE_DS_ADAM} \
                             ${CPU_EMBED} \
                             ${CPU_EMBED_FP32} \
                             ${HYBRID_ADAM_FLAG} \
                             --default_chunk_size=${CHUNK_SIZE} \
                             2>&1 | tee logs/log.${MODEL_NAME}_bs_${BS}_cs_${CS}_gpu_${GPU_NUM}
done
done
done
