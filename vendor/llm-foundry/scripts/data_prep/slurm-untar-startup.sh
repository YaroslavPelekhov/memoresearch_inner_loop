#!/bin/sh
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID
export GPUS_PER_NODE=$SLURM_GPUS_ON_NODE

echo "Hi from $(hostname)!"

export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/home/user/hpcx/ompi/lib:/home/user/hpcx/ucx/lib:/home/user/hpcx/ucc/lib:/home/user/hpcx/sharp/lib:/home/user/hpcx/nccl_rdma_sharp_plugin/lib:/home/user/hpcx/hcoll/lib:/home/user/hpcx/ompi/lib:/home/user/hpcx/ucx/lib:/home/user/hpcx/ucc/lib:/home/user/hpcx/sharp/lib:/home/user/hpcx/nccl_rdma_sharp_plugin/lib:/home/user/hpcx/hcoll/lib
export PATH=/home/user/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/home/user/hpcx/ompi/bin:/home/user/hpcx/ucx/bin:/home/user/hpcx/ucc/bin:/home/user/hpcx/sharp/bin:/home/user/hpcx/hcoll/bin:/home/user/hpcx/ompi/bin:/home/user/hpcx/ucx/bin:/home/user/hpcx/ucc/bin:/home/user/hpcx/sharp/bin:/home/user/hpcx/hcoll/bin

export ROOT_DIR=${ROOT_DIR:-/path/to/workdir/root}
export PYTHONPATH=${ROOT_DIR}:${ROOT_DIR}/contrib/composer:${ROOT_DIR}/contrib/streaming

LOG_DIR_NAME=${LOG_DIR_NAME:-logs_untar_papaya}
LOGDIR="${ROOT_DIR}/${LOG_DIR_NAME}/$SLURM_JOB_ID"
mkdir -p $LOGDIR
echo $(env) > $LOGDIR/$SLURM_PROCID.env

# ---

# Update PATH with the path to `parallel` binary directory if it's not present in the image.
# export PATH=/path/to/parallel/bin/dir:$PATH

# Read variables from the `sbatch` launch command.
SOURCE_DIR=$SOURCE_DIR
DEST_DIR=$DEST_DIR
TASK_FILE_PATH=$TASK_FILE_PATH
EXECUTOR=$SLURM_PROCID
N_PROC=$N_PROC

SOURCE_FILES=$(python3 << EOF
import json

fin = open('$TASK_FILE_PATH')
task_files = json.load(fin)[$EXECUTOR]
fin.close()

source_files = [source_file for source_file, _ in task_files]
print('\n'.join(source_files))
EOF
)

TARGET_FILES=$(python3 << EOF
import json

fin = open('$TASK_FILE_PATH')
task_files = json.load(fin)[$EXECUTOR]
fin.close()

target_files = [target_file for _, target_file in task_files]
print('\n'.join(target_files))
EOF
)

# Create destination directory if it does not exist.
if [ ! -d "$DEST_DIR" ]; then
  mkdir -p "$DEST_DIR"
fi


# Define uncompression function. It strips first components of the file path
# (we have absolute paths in archive) to extract files to the same relative
# directory structure in destination folder.
uncompress_file() {
  local file="$1"

  if [[ ! -f "$file" ]]; then
          exit
  fi

  # Strip first 6 components. Used for the numbered set processing (3.5, test_data).
  tar -xf "$file" -C "$DEST_DIR" --strip-components=6

  # Strip first 7 components. Used for processing specific dataset (3.5/wikipedia, test_data/).
  # tar -xf "$file" -C "$DEST_DIR" --strip-components=7
}

export -f uncompress_file

# Determine source and target files and get unprocessed files. Strip .tar.gz suffix from the
# source file and then append it back when we pass it to uncompress function.
source_files_relative=$(echo "$SOURCE_FILES" | sort | sed "s|^$SOURCE_DIR/||")
target_files_processed_relative=$(find $DEST_DIR -type f | sort | sed "s|^$DEST_DIR/||")

unprocessed_files=$(comm -23 <(echo "$source_files_relative" | sed "s/.tar.gz$//") <(echo "$target_files_processed_relative"))

echo "Unprocessed files: $unprocessed_files"
echo

echo "$(which parallel)"

echo "$unprocessed_files" | parallel -j $N_PROC --bar --no-notice uncompress_file "$SOURCE_DIR/{}.tar.gz"
