
export GPU_NUM=${GPU_NUM:-1}
export CS=${CS:-64}

export margin_use_ratio=${margin_use_ratio:-0.8}
# if warmup fails, lower the ratio
export warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio:-0.2}
export overall_gpu_mem_ratio=${overall_gpu_mem_ratio:-0.8}

let CHUNK_SIZE=${CS}*1024*1024
export PYTHONPATH=../:${PYTHONPATH}

python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} \
    test_loss_scale.py \
    --use_gpu_fp32_convert_for_adam \
    --overall_gpu_mem_ratio=${overall_gpu_mem_ratio} \
    --margin_use_ratio=${margin_use_ratio} \
    --warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio} \
    --default_chunk_size=${CHUNK_SIZE} \
    --use_deepspeed_cpu_adam \
    --use_hybrid_adam \
    --use_cpu_embedding \
    2>&1 | tee ./logs/log.loss_scale_gpu_${GPU_NUM}_cs_${CS}_bs_${BS}_cpueb_${CPU_EBD}_margin_${margin_use_ratio}_warmup_${warmup_gpu_chunk_mem_ratio}_gpu_${overall_gpu_mem_ratio}_adamcvt_${GPU_BOOST_ADAM}
