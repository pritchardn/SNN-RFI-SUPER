"""
This script is used to run the optuna optimization using MPI. Suitable for HPC environments.
"""

import json
import os

import optuna
from mpi4py import MPI
from optuna._callbacks import RetryFailedTrialCallback

from config import get_default_params
from experiment import Experiment

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def objective_mpi(trial):
    dataset = os.getenv("DATASET", "HERA")
    model_type = os.getenv("MODEL_TYPE", "FC_LATENCY")
    config = get_default_params(dataset, model_type)

    config["data_source"]["data_path"] = os.getenv("DATA_PATH", "./data")
    config["data_source"]["limit"] = float(os.getenv("LIMIT", 0.1))
    config["data_source"]["patch_size"] = int(os.getenv("PATCH_SIZE", 32))
    config["data_source"]["stride"] = int(os.getenv("STRIDE", 32))
    config["data_source"]["delta_normalization"] = (
        os.getenv("DELTA_NORMALIZATION", False) == "True"
    )

    config["dataset"]["batch_size"] = int(os.getenv("BATCH_SIZE", 36))
    config["model"]["num_hidden"] = trial.suggest_categorical(
        "num_hidden", [128, 256, 512]
    )
    config["model"]["num_layers"] = trial.suggest_int("num_layers", 2, 6)
    config["model"]["beta"] = trial.suggest_float("beta", 0.0, 1.0)

    config["trainer"]["epochs"] = int(os.getenv("EPOCHS", 100))

    config["encoder"]["method"] = os.getenv("ENCODER_METHOD", "LATENCY")
    config["encoder"]["exposure"] = trial.suggest_int("exposure", 1, 64)
    config["encoder"]["exposure_mode"] = os.getenv("FORWARD_EXPOSURE", "latency")

    print(json.dumps(config, indent=4))
    config = comm.bcast(config, root=0)
    MPI.Finalize()
    root_dir = os.getenv("OUTPUT_DIR", "./")
    experiment = Experiment(root_dir=root_dir)
    experiment.from_config(config)
    experiment.prepare()
    experiment.train()
    accuracy, mse, auroc, auprc, f1 = experiment.evaluate()
    return accuracy, mse, auroc, auprc, f1


def gremlin_objective_mpi():
    config = comm.bcast(None, root=0)
    MPI.Finalize()
    root_dir = os.getenv("OUTPUT_DIR", "./")
    experiment = Experiment(root_dir=root_dir)
    experiment.from_config(config)
    experiment.prepare()
    experiment.train()
    accuracy, mse, auroc, auprc, f1 = experiment.evaluate()
    return accuracy, mse, auroc, auprc, f1


def main_mpi():
    optuna_db = os.getenv("OPTUNA_DB", None)
    direction = ["maximize", "minimize", "maximize", "maximize", "maximize"]
    if optuna_db:
        storage = optuna.storages.RDBStorage(
            url=optuna_db,
            heartbeat_interval=60,
            grace_period=120,
        )
        study = optuna.create_study(
            study_name=os.getenv("STUDY_NAME"),
            storage=storage,
            load_if_exists=True,
            directions=direction,
        )
        study.optimize(objective_mpi, n_trials=1)
    else:
        study = optuna.create_study(
            study_name=os.getenv("STUDY_NAME"),
            directions=direction,
        )
        study.optimize(objective_mpi, n_trials=1)
        print("Study statistics")
        print("Number of finished trials: ", len(study.trials))
        print("  Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        with open("study.json", "w") as ofile:
            json.dump(study.best_trial.params, ofile, indent=4)


if __name__ == "__main__":
    if rank == 0:
        main_mpi()
    else:
        gremlin_objective_mpi()
