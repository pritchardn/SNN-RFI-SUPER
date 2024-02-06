import os
import json
import optuna
from optuna.trial import TrialState


def main(optuna_db):
    if optuna_db:
        storage = optuna.storages.RDBStorage(
            url=optuna_db,
        )
        print(f"Optuna DB: {optuna_db}")
        study = optuna.load_study(
            study_name=os.getenv("STUDY_NAME"), storage=storage
        )
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
        best_trials = study.best_trials
        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of complete trials: ", len(complete_trials))
        for trial in best_trials:
            print("  Best trial: ")
            print("    Value: ", trial.values)
            print("    Params: ")
            for key, value in trial.params.items():
                print(f"      {key}: {value}")
        with open("trials.json", "w") as ofile:
            completed_trials_out = []
            for trial_params in complete_trials:
                completed_trials_out.append(trial_params.params)
            json.dump(completed_trials_out, ofile, indent=4)
        with open("best_trial.json", "w") as ofile:
            best_trials_out = []
            for trial_params in best_trials:
                best_trials_out.append({
                    "params": trial_params.params,
                    "values": trial_params.values,
                })
            json.dump(best_trials_out, ofile, indent=4)
    else:
        raise ValueError("No optuna DB specified.")


if __name__ == "__main__":
    OPTUNA_DB = os.getenv("OPTUNA_DB", None)
    main(OPTUNA_DB)
