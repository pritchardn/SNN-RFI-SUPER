import optuna
from optuna.importance import get_param_importances
import os

optuna_db = os.getenv("OPTUNA_DB")


def main(experiment_list: list[str]):
    for study_name in experiment_list:
        study = optuna.load_study(study_name=study_name, storage=optuna_db)
        for i in range(4):
            importances = get_param_importances(study, target=lambda t: t.values[i])
            print(importances)


if __name__ == "__main__":
    experiment_list = [
        "SNN-SUPER-B-HERA-LATENCY-FC_LATENCY-100--True",
    ]
    main(experiment_list)
