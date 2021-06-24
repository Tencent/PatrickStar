GPU_NUM=1
# RES_CHECK_FLAG="--res_check"
RES_CHECK_FLAG=""
MAX_CPU_MEMORY=12884901888
MAX_GPU_MEMORY=4294967296
CHUNK_SIZE=39452672
python ../launcher/runner.py --num_nodes 1 --num_gpus ${GPU_NUM} test_bert.py ${RES_CHECK_FLAG} --use_ckp --use_fp16 --use_ps --use_fake_dist --max_cpu_memory=${MAX_CPU_MEMORY} --max_gpu_memory=${MAX_GPU_MEMORY} --use_cpu_embedding --default_chunk_size=${CHUNK_SIZE}  2>&1 | tee log
