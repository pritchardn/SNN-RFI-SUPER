#!/bin/bash
#SBATCH --job-name=SNN-SUPER
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-0
#SBATCH --partition=work
#SBATCH --account=pawsey0411-gpu

module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-RFI-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OPTUNA_DB="postgresql://${DBUSR}:${DBPW}@${DBHOST}:5432"
export STUDY_NAME="SNN-SUPER-HERA-LATENCY-10"
export ROOT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/hera"

srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 python3 main_hpc.py
