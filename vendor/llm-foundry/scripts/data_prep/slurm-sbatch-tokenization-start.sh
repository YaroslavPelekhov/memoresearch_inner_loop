#!/bin/bash
#SBATCH --job-name=tokenization_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/path/to/repo/root/scripts/data_prep/logs/%j/log.log
#SBATCH --error=/path/to/repo/root/scripts/data_prep/logs/%j/log.err
# #SBATCH --nodelist=gpu[29-33,35-36,38-43,45-64]


set -evx

WORKDIR="/path/to/workdir/repo/root"
STARTUP_SCRIPT=${STARTUP_SCRIPT:-$WORKDIR/scripts/data_prep/slurm-tokenization-startup.sh}
MOUNTS=${SLURM_CONTAINER_MOUNTS:-"/path/to/mounts:/path/to/mounts"}
CONTAINER_IMAGE=${SLURM_CONTAINER_IMAGE:-/path/to/image.sqsh}


# download images to nodes
srun --container-image="$CONTAINER_IMAGE" \
     --no-container-mount-home \
     --container-name="$(basename $CONTAINER_IMAGE)" \
     --container-mounts="$MOUNTS" \
     --container-workdir="$WORKDIR" \
     true

set +e

# run scripts
srun --container-name="$(basename $CONTAINER_IMAGE)" \
     --no-container-mount-home \
     --container-mounts="$MOUNTS" \
     --container-workdir="$WORKDIR" \
     -K \
     bash $STARTUP_SCRIPT

exitcode=$?
set -e
echo "job exited with code $exitcode"
