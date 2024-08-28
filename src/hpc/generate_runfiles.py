"""
This script generates runfiles for supercomputer use.
"""

import os

models = {
    "FC": [
        ("FC_LATENCY", "LATENCY"),
        ("FC_RATE", "RATE"),
        ("FC_DELTA", "DELTA"),
        ("FC_FORWARD_STEP", "FORWARDSTEP"),
        ("FC_DELTA_EXPOSURE", "DELTA_EXPOSURE"),
    ],
    "RNN": [
        ("RNN_LATENCY", "LATENCY"),
        ("RNN_RATE", "RATE"),
        ("RNN_DELTA", "DELTA"),
        ("RNN_FORWARD_STEP", "FORWARDSTEP"),
    ],
}
datasets = ["HERA", "LOFAR", "TABASCAL"]
forwardstep_exposures = ["direct", "first", "latency"]
delta_normalization = [True, False]


def prepare_singlerun(
    model,
    encoding,
    dataset,
    forward_step_exposure="None",
    delta_norm=False,
    num_nodes=1,
):
    forward_step_directory = (
        '''/${FORWARD_EXPOSURE}"''' if forward_step_exposure != "None" else '"'
    )
    runfiletext = (
        f"""#!/bin/bash
#SBATCH --job-name=SNN-SUPER-{model}-{encoding}-{dataset}
#SBATCH --nodes={num_nodes}
#SBATCH --time=24:00:00
#SBATCH --mem={"230" if dataset == "LOFAR" else "115"}G 
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-9
#SBATCH --partition=work
#SBATCH --account=pawsey0411

export DATASET="{dataset}"
export LIMIT="1.0"
export MODEL_TYPE="{model}"
export ENCODER_METHOD="{encoding}"
export FORWARD_EXPOSURE="{forward_step_exposure}"
export NNODES="{num_nodes}"
export DELTA_NORMALIZATION="{delta_norm}"

module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/${{MODEL_TYPE}}/${{ENCODER_METHOD}}/${{DATASET}}/${{DELTA_NORMALIZATION}}/${{NUM_HIDDEN}}/${{LIMIT}}"""
        + forward_step_directory
        + """
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1
export OMP_PLACES=cores     
export OMP_PROC_BIND=close  
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
   
srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS -m block:block:block python3 main.py
    """
    )
    return runfiletext


def prepare_optuna(
    model,
    encoding,
    dataset,
    limit,
    forward_step_exposure="None",
    delta_norm=False,
    num_nodes=1,
):
    forward_step_directory = (
        '''/${FORWARD_EXPOSURE}"''' if forward_step_exposure != "None" else '"'
    )
    study_name = (
        f"""export STUDY_NAME="SNN-SUPER-B-${{DATASET}}-${{ENCODER_METHOD}}-${{MODEL_TYPE}}-{limit}-${{NUM_HIDDEN}}"""
        + ("""-${FORWARD_EXPOSURE}""" if forward_step_exposure != "None" else """""")
        + '-${DELTA_NORMALIZATION}"'
    )
    runfiletext = (
        f"""#!/bin/bash
#SBATCH --job-name=SNN-SUPER-{model}-{encoding}-{dataset}
#SBATCH --nodes={num_nodes}
#SBATCH --time=24:00:00
#SBATCH --mem={"230" if dataset == "LOFAR" else "115"}G 
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --output=super_%A_%a.out
#SBATCH --error=super_%A_%a.err
#SBATCH --array=0-49%4
#SBATCH --partition=work
#SBATCH --account=pawsey0411

export DATASET="{dataset}"
export LIMIT="{limit / 100.0}"
export MODEL_TYPE="{model}"
export ENCODER_METHOD="{encoding}"
export FORWARD_EXPOSURE="{forward_step_exposure}"
export NNODES="{num_nodes}"
export DELTA_NORMALIZATION="{delta_norm}"


module load python/3.10.10

cd /software/projects/pawsey0411/npritchard/setonix/2023.08/python/SNN-SUPER/src
source /software/projects/pawsey0411/npritchard/setonix/2023.08/python/snn-nln/bin/activate

export DATA_PATH="/scratch/pawsey0411/npritchard/data"
export OPTUNA_DB=${{OPTUNA_URL}} # Need to change on super-computer before submitting\n"""
        + study_name
        + """
export OUTPUT_DIR="/scratch/pawsey0411/npritchard/outputs/snn-super/optuna/${MODEL_TYPE}/${ENCODER_METHOD}/${DATASET}/${DELTA_NORMALIZATION}/${NUM_HIDDEN}/${LIMIT}"""
        + forward_step_directory
        + """
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)
export MPICH_OFI_STARTUP_CONNECT=1
export MPICH_OFI_VERBOSE=1
export OMP_PLACES=cores     
export OMP_PROC_BIND=close  
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

srun -N $SLURM_JOB_NUM_NODES -n $SLURM_NTASKS -c $OMP_NUM_THREADS -m block:block:block python3 optuna_main_mpi.py
"""
    )
    return runfiletext


def write_bashfile(out_dir, name, runfiletext):
    with open(os.path.join(out_dir, f"{name}.sh"), "w") as f:
        f.write(runfiletext)


def write_runfiles(out_dir, model, encoding, dataset, num_nodes, delta_norm):
    if encoding == "FORWARDSTEP":
        for forward_step_exposure in forwardstep_exposures:
            write_bashfile(
                out_dir,
                f"{dataset}-{encoding}-{forward_step_exposure}",
                prepare_singlerun(
                    model,
                    encoding,
                    dataset,
                    forward_step_exposure,
                    delta_norm=delta_norm,
                    num_nodes=num_nodes,
                ),
            )
    else:
        write_bashfile(
            out_dir,
            f"{dataset}-{encoding}",
            prepare_singlerun(
                model,
                encoding,
                dataset,
                delta_norm=delta_norm,
                num_nodes=num_nodes,
            ),
        )
    limit = 10
    if encoding == "FORWARDSTEP":
        for forward_step_exposure in forwardstep_exposures:
            write_bashfile(
                out_dir,
                f"optuna-{dataset}-{encoding}-{limit}-{forward_step_exposure}",
                prepare_optuna(
                    model,
                    encoding,
                    dataset,
                    limit,
                    forward_step_exposure,
                    delta_norm=delta_norm,
                    num_nodes=num_nodes,
                ),
            )
    else:
        write_bashfile(
            out_dir,
            f"optuna-{dataset}-{encoding}-{limit}",
            prepare_optuna(
                model,
                encoding,
                dataset,
                limit,
                delta_norm=delta_norm,
                num_nodes=num_nodes,
            ),
        )
    limit = 100
    if dataset == "LOFAR":
        limit = 50
    if encoding == "FORWARDSTEP":
        for forward_step_exposure in forwardstep_exposures:
            write_bashfile(
                out_dir,
                f"optuna-{dataset}-{encoding}-{limit}-{forward_step_exposure}",
                prepare_optuna(
                    model,
                    encoding,
                    dataset,
                    limit,
                    forward_step_exposure,
                    delta_norm=delta_norm,
                    num_nodes=num_nodes,
                ),
            )
    else:
        write_bashfile(
            out_dir,
            f"optuna-{dataset}-{encoding}-{limit}",
            prepare_optuna(
                model,
                encoding,
                dataset,
                limit,
                delta_norm=delta_norm,
                num_nodes=num_nodes,
            ),
        )


def main(out_dir, num_nodes):
    for major_model, minor_models in models.items():
        for model, encoding in minor_models:
            for dataset in datasets:
                for delta_norm in delta_normalization:
                    out_dir_temp = os.path.join(
                        out_dir,
                        major_model,
                        encoding,
                        dataset,
                        "DELTA_NORM" if delta_norm else "ORIGINAL",
                    )
                    os.makedirs(out_dir_temp, exist_ok=True)
                    write_runfiles(
                        out_dir_temp,
                        model,
                        encoding,
                        dataset,
                        num_nodes,
                        delta_norm,
                    )


if __name__ == "__main__":
    main(f".{os.sep}src{os.sep}hpc{os.sep}pawsey", 8)
