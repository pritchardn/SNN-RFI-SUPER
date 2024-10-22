"""
This script is used to extract the best trials from an optuna study and save them to a JSON file.
"""

import json
import os

import optuna
from optuna.trial import TrialState
from tqdm import tqdm


def prepare_trial_json(trial, metric_names):
    out_vals = {}
    for name, val in zip(metric_names, trial.values):
        out_vals[name] = val
    return {"params": trial.params, "values": out_vals, "id": trial.number}


def main(optuna_db):
    if optuna_db:
        storage = optuna.storages.RDBStorage(
            url=optuna_db,
        )
        study_name = os.getenv("STUDY_NAME")
        metric_names = ["accuracy", "mse", "auroc", "auprc", "f1"]
        print(f"Optuna DB: {optuna_db}")
        study = optuna.load_study(study_name=study_name, storage=storage)
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        best_trials = study.best_trials
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of complete trials: ", len(complete_trials))
        for trial in best_trials:
            print(json.dumps(prepare_trial_json(trial, metric_names), indent=4))
        with open(f"{study_name}_trials.json", "w") as ofile:
            completed_trials_out = []
            for trial_params in complete_trials:
                completed_trials_out.append(
                    prepare_trial_json(trial_params, metric_names)
                )
            json.dump(completed_trials_out, ofile, indent=4)
        with open(f"{study_name}_best_trial.json", "w") as ofile:
            best_trials_out = []
            for trial_params in best_trials:
                best_trials_out.append(prepare_trial_json(trial_params, metric_names))
            json.dump(best_trials_out, ofile, indent=4)
    else:
        raise ValueError("No optuna DB specified.")


if __name__ == "__main__":
    OPTUNA_DB = os.getenv("OPTUNA_DB", None)
    experiment_list = [
        "SNN-SUPER-B-HERA-LATENCY-FC_LATENCY-100--False",
        "SNN-SUPER-B-HERA-LATENCY-FC_LATENCY-100--True",
        "SNN-SUPER-B-HERA-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-100--False",
        "SNN-SUPER-B-HERA-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-100--True-True",
        "SNN-SUPER-B-HERA-FORWARDSTEP-FC_FORWARD_STEP-100--direct-False",
        "SNN-SUPER-B-HERA-FORWARDSTEP-FC_FORWARD_STEP-100--direct-True",
        "SNN-SUPER-B-HERA-ANN-ANN-100-False",
        "SNN-SUPER-B-HERA-ANN-ANN-100-True",
        "SNN-SUPER-C-LOFAR-FORWARDSTEP-FC_FORWARD_STEP-15--direct-True",
        "SNN-SUPER-C-LOFAR-FORWARDSTEP-FC_FORWARD_STEP-15--direct-False",
        "SNN-SUPER-C-LOFAR-LATENCY-FC_LATENCY-15--True",
        "SNN-SUPER-C-LOFAR-LATENCY-FC_LATENCY-15--False",
        "SNN-SUPER-C-LOFAR-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-15--True",
        "SNN-SUPER-C-LOFAR-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-15--False",
        "SNN-SUPER-C-LOFAR-ANN-FC_ANN-15--True",
        "SNN-SUPER-C-LOFAR-ANN-FC_ANN-15--False",
    ]
    for experiment_name in tqdm(experiment_list):
        os.environ["STUDY_NAME"] = experiment_name
        main(OPTUNA_DB)
