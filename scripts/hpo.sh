#!/bin/bash
#SBATCH -C gpu
#SBATCH -J hpo
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH -c 10
#SBATCH -t 8:00:00
#SBATCH -o logs/%x-%j.out

. scripts/setup_cgpu.sh

python hpo.py
