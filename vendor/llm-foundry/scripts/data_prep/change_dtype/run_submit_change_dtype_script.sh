BASE_IMAGE="cr.msk.sbercloud.ru/19d74875-55e7-4f05-92d7-5bff1b43b8a0/llm-foundry-train:v000_cuda121_torch211_nccl2205_hpcx"

DATASET_DIR=""
SHARDS_PATH=${DATASET_DIR/datasets/datasets_reduce_dtype}
CONVERATION_OUTPUT_PATH="${SHARDS_PATH}_reduced"
SCRIPT="change_mds_dataset_dtype.py"
SCRIPT_ADDITIONAL_PARAMS=""
TASK_NAME="reduce_dtype_<DESC>"
NUM_EXECUTORS=2
N_GPUS=8

echo "DATASET_DIR: $DATASET_DIR"
echo "SHARDS_PATH: $SHARDS_PATH"
echo "CONVERATION_OUTPUT_PATH: $CONVERATION_OUTPUT_PATH"

python3 unmerge_shards.py \
  --input-dir $DATASET_DIR \
  --output-dir $SHARDS_PATH

_PARAMS=""
if [ ! -z "$SCRIPT_ADDITIONAL_PARAMS" ]; then
  _PARAMS="${_PARAMS} --script-additional-params ${SCRIPT_ADDITIONAL_PARAMS}"
fi

python submit_change_dtype_job.py $_PARAMS \
    --base-image $BASE_IMAGE \
    --input-path $SHARDS_PATH \
    --output-path $CONVERATION_OUTPUT_PATH \
    --task-name "$TASK_NAME" \
    --script $SCRIPT \
    --num-executors $NUM_EXECUTORS \
    --n-gpus $N_GPUS \
    --repair
