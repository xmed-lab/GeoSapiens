cd ../../../..

###--------------------------------------------------------------
# DEVICES=0,
DEVICES=1,

RUN_FILE='./tools/dist_test.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

DATASET='teeth'
MODEL="lora_sapiens_0.3b-210e_${DATASET}-1024x768"
# MODEL="sapiens_0.3b-210e_${DATASET}-1024x768"
TEST_BATCH_SIZE_PER_GPU=32

CHECKPOINT="${GeoSapiensBase}/pretrain/checkpoints/sapiens_0.3b/epoch_210.pth" 

mode='debug'




BASE_PATH="${GeoSapiensBase}"
cd "${BASE_PATH}/pose"
CONFIG_FILE="${BASE_PATH}/pose/configs/GeoSapiens/${DATASET}/${MODEL}.py"
OUTPUT_DIR="${BASE_PATH}/pose/Outputs/test/${DATASET}/${MODEL}/node"
OUTPUT_DIR="$(echo "${OUTPUT_DIR}/$(date +"%m-%d-%Y_%H:%M:%S")")"

export TF_CPP_MIN_LOG_LEVEL=2

## set the options for the test
OPTIONS="$(echo "test_dataloader.batch_size=$TEST_BATCH_SIZE_PER_GPU")"

##--------------------------------------------------------------
## if mode is multi-gpu, then run the following
## else run the debugging on a single gpu
if [ "$mode" = "debug" ]; then
    TEST_BATCH_SIZE_PER_GPU=16 ## works for single gpu

    OPTIONS="$(echo "test_dataloader.batch_size=${TEST_BATCH_SIZE_PER_GPU} test_dataloader.num_workers=0 test_dataloader.persistent_workers=False")"
    CUDA_VISIBLE_DEVICES=${DEVICES} python ${BASE_PATH}/pose/tools/test.py ${CONFIG_FILE} ${CHECKPOINT} --work-dir ${OUTPUT_DIR} --cfg-options ${OPTIONS} --dump ${OUTPUT_DIR}/dump.pkl "$@"

elif [ "$mode" = "multi-gpu" ]; then
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))

    LOG_FILE="$(echo "${OUTPUT_DIR}/log.txt")"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} ${CHECKPOINT}\
            ${NUM_GPUS} \
            --work-dir ${OUTPUT_DIR} \
            --cfg-options ${OPTIONS} \
            | tee ${LOG_FILE}

fi
