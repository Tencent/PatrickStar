export MODEL_NAME=""
export BS=12
export CS=384
export CPU_EBD=0
export SP=0
export ACT_OFFLOAD=0
export NO_RETRY=0
export SKIP_LOG_EXSIT=0
export MSC=1
export CACHE=1
export GPU_NUM=8
export MODEL_TYPE="BERT"


for GPU_NUM in 8
do
for MODEL_NAME in "GPT_DS_40B"
do
for BS in 4 
do
for CS in 288 
do
for CPU_EBD in 0
do
for SP in 0
do
for ACT_OFFLOAD in 0
do
for MSC in 1
do
for CACHE in 1
do
echo "****************** Begin ***************************"
echo "* benchmarking CS ${CS} BS ${BS} MODEL ${MODEL_NAME} "
echo "* CPU_EBD ${CPU_EBD} SP ${SP} ACT_OFFLOAD ${ACT_OFFLOAD} MSC ${MSC} CACHE ${CACHE}"
bash ../run_transformers.sh
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
