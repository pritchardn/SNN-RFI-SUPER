import json
import os

import optuna
from optuna.trial import TrialState


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
    main(OPTUNA_DB)
