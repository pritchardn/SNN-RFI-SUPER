#!/bin/bash
#SBATCH --job-name=SNN-SUPER-FC_LATENCY-LATENCY-HERA-512
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-100%4
#SBATCH --partition=gpu
#SBATCH --account=pawsey0411-gpu

export DATASET="HERA"
export LIMIT="1.0"
export MODEL_TYPE="FC_LATENCY"
export ENCODER_METHOD="LATENCY"
export NUM_HIDDEN="512"

module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OPTUNA_DB=${OPTUNA_URL} # Need to change on super-computer before submitting
export STUDY_NAME="SNN-SUPER-${DATASET}-${ENCODER_METHOD}-100-${NUM_HIDDEN}"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/FC/${ENCODER_METHOD}/${DATASET}/${NUM_HIDDEN}/${LIMIT}"
export MPICH_GPU_SUPPORT_ENABLED=1

srun -N 1 -n 1 -c 64 --gres=gpu:8 --gpus-per-task=8 python3 optuna_main.py
