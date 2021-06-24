GPU_NUM=1
RES_CHECK_FLAG="--res_check"
# RES_CHECK_FLAG=""
let MAX_CPU_MEMORY=12*1024*1024*1024
let MAX_GPU_MEMORY=4*1024*1024*1024
let CHUNK_SIZE=32*1024*1024

python ../launcher/runner.py --num_nodes 1 \
                             --num_gpus ${GPU_NUM} \
                             test_bert.py ${RES_CHECK_FLAG} \
                             --use_ckp \
                             --use_fp16 \
                             --use_ps \
                             --use_fake_dist \
                             --max_cpu_memory=${MAX_CPU_MEMORY} \
                             --max_gpu_memory=${MAX_GPU_MEMORY} \
                             --use_cpu_embedding \
                             --cpu_embedding_fp32 \
                             --default_chunk_size=${CHUNK_SIZE} \
                             2>&1 | tee log
