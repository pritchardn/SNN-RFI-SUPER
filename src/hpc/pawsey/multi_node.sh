#!/bin/bash
#SBATCH --job-name=SNN-SUPER-DDP
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-0
#SBATCH --partition=gpu
#SBATCH --account=pawsey0411-gpu
#SBATCH --ntasks-per-node=8

module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/hera"
export MPICH_GPU_SUPPORT_ENABLED=1
export NNODES=2

srun -N 2 -n 8 -c 8 --gres=gpu:8 python3 main.py