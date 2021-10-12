export GPU_NUM=${GPU_NUM:-4}
export MODEL_NAME="Bertlarge"
export CS=32
export BS=8

export margin_use_ratio=${margin_use_ratio:-0.8}
# if warmup fails, lower the ratio
export warmup_gpu_chunk_mem_ratio=${warmup_gpu_chunk_mem_ratio:-0.2}
export overall_gpu_mem_ratio=${overall_gpu_mem_ratio:-0.8}

# for MODEL_NAME in "GPT3_8B" "GPT3_10B" "GPT3_12B" "GPT3_13B"
for MODEL_NAME in "GPT3_6B"
do
for BS in 32 16 8
do
for CS in 64 48 32 128
do
for CPU_EBD in 1 0
do
for AW in 0 1
do
for ACT_OFFLOAD in 0 1
do
let CHUNK_SIZE=${CS}*1024*1024
echo "benchmarking ${CS} ${BS} ${MODEL_NAME} ${CPU_EBD} ${AW} ${ACT_OFFLOAD}"
env RELEASE_AFTER_INIT=0 bash run_bert.sh
done
done
done
done
done
done
