export MODEL_NAME=""
export BS=32
export CS=64
export CPU_EBD=1
export SP=0
export ACT_OFFLOAD=0
export NO_RETRY=0
export SKIP_LOG_EXSIT=1
export MSC=1
export CACHE=1
export GPU_NUM=1


for GPU_NUM in 1 2 4 8
do
for MODEL_NAME in "GPT_DS_20B" "GPT_DS_40B"
do
for BS in 8 4 16
do
for CS in 256 384
do
for CPU_EBD in 0
do
for SP in 0
do
for ACT_OFFLOAD in 0
do
for MSC in 0
do
for CACHE in 0 1
do
echo "****************** Begin ***************************"
echo "* benchmarking CS ${CS} BS ${BS} MODEL ${MODEL_NAME} "
echo "* CPU_EBD ${CPU_EBD} SP ${SP} ACT_OFFLOAD ${ACT_OFFLOAD} MSC ${MSC} CACHE ${CACHE}"
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
done
done
done
