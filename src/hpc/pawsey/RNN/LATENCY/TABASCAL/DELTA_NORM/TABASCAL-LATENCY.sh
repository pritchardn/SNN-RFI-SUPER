#!/bin/bash
#SBATCH --job-name=SNN-SUPER-RNN_LATENCY-LATENCY-TABASCAL
#SBATCH --nodes=8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-9
#SBATCH --partition=gpu
#SBATCH --account=pawsey0411-gpu

export DATASET="TABASCAL"
export LIMIT="1.0"
export MODEL_TYPE="RNN_LATENCY"
export ENCODER_METHOD="LATENCY"
export FORWARD_EXPOSURE="None"
export NNODES="8"
export DELTA_NORMALIZATION="True"

module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/${MODEL_TYPE}/${ENCODER_METHOD}/${DATASET}/${DELTA_NORMALIZATION}/${NUM_HIDDEN}/${LIMIT}"
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_PLACES=cores     
export OMP_PROC_BIND=close  
   
srun -N 1 -n 1 -c 64 --gpus-per-task=8 --gpu-bind=closest python3 main.py
    