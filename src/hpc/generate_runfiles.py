import os

models = {"FC": [
    ("FC_LATENCY", "LATENCY"), ("FC_RATE", "RATE"), ("FC_DELTA", "DELTA"),
    ("FC_FORWARD_STEP", "FORWARDSTEP")
]}
datasets = ["HERA", "LOFAR", "TABASCAL"]


def prepare_singlerun(model, encoding, dataset, size):
    runfiletext = f"""#!/bin/bash
#SBATCH --job-name=SNN-SUPER-{model}-{encoding}-{dataset}-{size}
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-9
#SBATCH --partition=gpu
#SBATCH --account=pawsey0411-gpu

export DATASET="{dataset}"
export LIMIT="1.0"
export MODEL_TYPE="{model}"
export ENCODER_METHOD="{encoding}"
export NUM_HIDDEN="{size}"

module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/FC/${{ENCODER_METHOD}}/${{DATASET}}/${{NUM_HIDDEN}}/${{LIMIT}}"
export MPICH_GPU_SUPPORT_ENABLED=1
   
srun -N 1 -n 1 -c 64 --gres=gpu:8 --gpus-per-task=8 python3 main.py
    """
    return runfiletext


def prepare_optuna(model, encoding, dataset, size, limit):
    runfiletext = f"""#!/bin/bash
#SBATCH --job-name=SNN-SUPER-{model}-{encoding}-{dataset}-{size}
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-100%4
#SBATCH --partition=gpu
#SBATCH --account=pawsey0411-gpu

export DATASET="{dataset}"
export LIMIT="{limit/100.0}"
export MODEL_TYPE="{model}"
export ENCODER_METHOD="{encoding}"
export NUM_HIDDEN="{size}"

module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OPTUNA_DB=${{OPTUNA_URL}} # Need to change on super-computer before submitting
export STUDY_NAME="SNN-SUPER-${{DATASET}}-${{ENCODER_METHOD}}-{limit}-${{NUM_HIDDEN}}"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/FC/${{ENCODER_METHOD}}/${{DATASET}}/${{NUM_HIDDEN}}/${{LIMIT}}"
export MPICH_GPU_SUPPORT_ENABLED=1

srun -N 1 -n 1 -c 64 --gres=gpu:8 --gpus-per-task=8 python3 optuna_main.py
"""
    return runfiletext


def write_bashfile(out_dir, name, runfiletext):
    with open(os.path.join(out_dir, f"{name}.sh"), "w") as f:
        f.write(runfiletext)


def write_runfiles(out_dir, model, encoding, dataset, size):
    write_bashfile(out_dir, f"{dataset}-{encoding}-{size}",
                   prepare_singlerun(model, encoding, dataset, size))
    limit = 10
    write_bashfile(out_dir, f"optuna-{dataset}-{encoding}-{size}-{limit}",
                   prepare_optuna(model, encoding, dataset, size, limit))
    limit = 100
    write_bashfile(out_dir, f"optuna-{dataset}-{encoding}-{size}-{limit}",
                   prepare_optuna(model, encoding, dataset, size, limit))


def main(out_dir):
    for major_model, minor_models in models.items():
        for model, encoding in minor_models:
            for dataset in datasets:
                for size in [128, 256, 512]:
                    out_dir_temp = os.path.join(out_dir, major_model, encoding, dataset,
                                                str(size))
                    os.makedirs(out_dir_temp, exist_ok=True)
                    write_runfiles(out_dir_temp, model, encoding, dataset, size)


if __name__ == "__main__":
    main(f".{os.sep}src{os.sep}hpc{os.sep}pawsey")