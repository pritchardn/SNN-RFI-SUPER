#!/bin/bash
#SBATCH --job-name=SNN-SUPER-FC_FORWARD_STEP-FORWARDSTEP-TABASCAL-256
#SBATCH --nodes=8
#SBATCH --time=24:00:00
#SBATCH --mem=115G
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-100%4
#SBATCH --partition=work
#SBATCH --account=pawsey0411

export DATASET="TABASCAL"
export LIMIT="0.1"
export MODEL_TYPE="FC_FORWARD_STEP"
export ENCODER_METHOD="FORWARDSTEP"
export NUM_HIDDEN="256"
export NUM_LAYERS="2"
export FORWARD_EXPOSURE="first"
export NNODES="8"


module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OPTUNA_DB=${OPTUNA_URL} # Need to change on super-computer before submitting
export STUDY_NAME="SNN-SUPER-${DATASET}-${ENCODER_METHOD}-10-${NUM_HIDDEN}-${FORWARD_EXPOSURE}"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/optuna/${MODEL_TYPE}/${ENCODER_METHOD}/${DATASET}/${NUM_LAYERS}/${NUM_HIDDEN}/${LIMIT}/${FORWARD_EXPOSURE}"
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1
export OMP_PLACES=cores     
export OMP_PROC_BIND=close  
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS -m block:block:block python3 optuna_main_mpi.py
