#!/bin/bash
#SBATCH --job-name=untar_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=/path/to/repo/root/scripts/data_prep/logs/logs_large%j.log
#SBATCH --error=/path/to/repo/root/scripts/data_prep/logs/logs_large%j.err
# #SBATCH --nodelist=gpu[01-08]

WORKDIR="/path/to/workdir/repo/root"
STARTUP_SCRIPT=${STARTUP_SCRIPT:-$WORKDIR/scripts/data_prep/slurm-untar-startup.sh}
MOUNTS=${SLURM_CONTAINER_MOUNTS:-"/path/to/mounts:/path/to/mounts"}
CONTAINER_IMAGE=${SLURM_CONTAINER_IMAGE:-/path/to/image.sqsh}

srun    --container-image=$CONTAINER_IMAGE \
        --no-container-mount-home \
        --container-mounts=$MOUNTS \
        --container-workdir=$WORKDIR \
        bash $STARTUP_SCRIPT
