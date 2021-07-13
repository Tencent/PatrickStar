GPU_NUM=1
# Check result correctness
# RES_CHECK_FLAG="--res_check"
# Use a single GPU card to simulate multiple-GPU training.
# FAKE_DIST="--use_fake_dist"


let CHUNK_SIZE=32*1024*1024

export PYTHONPATH=../:${PYTHONPATH}

export HYBRID_ADAM_FLAG="--use_hybrid_adam"
export USE_DS_ADAM="--use_deepspeed_cpu_adam"
export CPU_EMBED="--use_cpu_embedding"
export CPU_EMBED_FP32="--cpu_embedding_fp32"
export MODEL_NAME="GPTsmall"
python ../patrickstar/launcher/runner.py --num_nodes 1 \
                             --num_gpus ${GPU_NUM} \
                             test_bert.py ${RES_CHECK_FLAG} \
                             --use_ckp \
                             --use_fp16 \
                             --use_ps \
                             --model_name=${MODEL_NAME} \
                             ${FAKE_DIST} \
                             ${USE_DS_ADAM} \
                             ${CPU_EMBED} \
                             ${CPU_EMBED_FP32} \
                             ${HYBRID_ADAM_FLAG} \
                             --default_chunk_size=${CHUNK_SIZE} \
                             2>&1 | tee log
