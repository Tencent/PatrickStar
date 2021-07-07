GPU_NUM=1
# Check result correctness
# RES_CHECK_FLAG="--res_check"
# Use a single GPU card to simulate multiple-GPU training.
# FAKE_DIST="--use_fake_dist"
# let MAX_CPU_MEMORY=12*1024*1024*1024
# let MAX_GPU_MEMORY=4*1024*1024*1024

# HYBRID_ADAM_FLAG="--use_hybrid_adam"
let CHUNK_SIZE=32*1024*1024

export PYTHONPATH=../:${PYTHONPATH}

USE_DS_ADAM="--use_deepspeed_cpu_adam"

python ../patrickstar/launcher/runner.py --num_nodes 1 \
                             --num_gpus ${GPU_NUM} \
                             test_bert.py ${RES_CHECK_FLAG} \
                             --use_ckp \
                             --use_fp16 \
                             --use_ps \
                             ${FAKE_DIST} \
                             ${USE_DS_ADAM} \
                             --use_cpu_embedding \
                             --cpu_embedding_fp32 \
                             --use_deepspeed_cpu_adam \
                             ${HYBRID_ADAM_FLAG} \
                             --default_chunk_size=${CHUNK_SIZE} \
                             2>&1 | tee log
