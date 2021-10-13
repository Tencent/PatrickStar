mkdir -p ./logs

export MODEL_NAME=""
export BS=32
export CS=64
export CPU_EBD=1
export AW=0
export ACT_OFFLOAD=0
export NO_RETRY=1
export SKIP_LOG_EXSIT=1

for MODEL_NAME in "GPT2small"
do
for BS in 32
do
for CS in 64
do
for CPU_EBD in 1
do
for AW in 0
do
for ACT_OFFLOAD in 0 1
do
echo "****************** Begin ***************************"
echo "* benchmarking CS ${CS} BS ${BS} MODEL ${MODEL_NAME} "
echo "* CPU_EBD ${CPU_EBD} AW ${AW} ACT_OFFLOAD ${ACT_OFFLOAD}"
bash ../run_bert.sh
echo "****************** Finished ***************************"
echo ""
echo ""
done
done
done
done
done
done
