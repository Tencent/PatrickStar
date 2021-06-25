GPU_NUM=2
CHUNK_SIZE=50

let MAX_CPU_MEMORY=CHUNK_SIZE*2*14*6
let MAX_GPU_MEMORY=CHUNK_SIZE*2*GPU_NUM

python ../launcher/runner.py --num_nodes 1 --num_gpus ${GPU_NUM} dpp_test.py \
              --use_fake_dist \
              --max_cpu_memory=${MAX_CPU_MEMORY} \
              --max_gpu_memory=${MAX_GPU_MEMORY} \
              --use_cpu_embedding \
              --cpu_embedding_fp32 \
              --default_chunk_size=${CHUNK_SIZE} \
              2>&1 | tee log
