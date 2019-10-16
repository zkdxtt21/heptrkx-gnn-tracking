#!/bin/bash
#SBATCH -J train-cori
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 30
#SBATCH -o logs/%x-%j.out

# This is a generic script for submitting training jobs to Cori.
# You need to supply the config file with this script.

# Setup
mkdir -p logs
. scripts/setup_cori.sh

# Run training
srun -l -u python train.py -d ddp-mpi $@

# Run training with pytorch bottleneck profiler
#srun python -m torch.utils.bottleneck train.py $@
