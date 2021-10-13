mkdir -p ./logs

for MODEL_NAME in "GPT3_8B" "GPT3_10B" "GPT3_12B" "GPT3_11B"
do
for BS in 32 16 8
do
for CS in 64 128
do
for CPU_EBD in 1 0
do
for AW in 0 1
do
for ACT_OFFLOAD in 0 1
do
echo "benchmarking CS ${CS} BS ${BS} MODEL ${MODEL_NAME} "
echo "CPU_EBD ${CPU_EBD} AW ${AW} ACT_OFFLOAD ${ACT_OFFLOAD}"
# bash run_bert.sh
bash test.sh
done
done
done
done
done
done
