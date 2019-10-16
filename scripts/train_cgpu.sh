#!/bin/bash
#SBATCH -J train-cgpu
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -t 8:00:00
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH -d singleton
#SBATCH -o logs/%x-%j.out

# This is a generic script for submitting training jobs to Cori-GPU.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
. scripts/setup_cgpu.sh

# Single GPU training
#srun -u python train.py $@

# Multi-GPU training
srun -u -l --ntasks-per-node 8 \
    python train.py --rank-gpu -d ddp-file $@
