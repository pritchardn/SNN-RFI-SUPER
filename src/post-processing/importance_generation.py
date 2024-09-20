import csv
import json
import os
from collections import OrderedDict

import optuna
from optuna.importance import get_param_importances
from tqdm import tqdm

OPTUNA_DB = os.getenv("OPTUNA_DB")
OBJECTIVES = ["accuracy", "mse", "auroc", "auprc", "f1"]


def get_trial_importances(study: optuna.study.study):
    output = {}
    for i in range(len(OBJECTIVES)):
        importances = get_param_importances(study, target=lambda t: t.values[i])
        output[OBJECTIVES[i]] = importances
    return output


def write_importances_json(importances: dict, study_name: str):
    with open(f"{study_name}_importances.json", "w") as ofile:
        json.dump(importances, ofile, indent=4)


def write_importances_csv(importances: dict, study_name: str):
    with open(f"{study_name}_importances.csv", "w", newline='') as ofile:
        writer = csv.writer(ofile)
        writer.writerow(["metric", "beta", "exposure", "num_layers", "num_hidden"])
        for metric, param_importances in importances.items():
            outs = {"beta": 0.0, "exposure": 0.0, "num_layers": 0, "num_hidden": 0}
            outs.update(param_importances)
            writer.writerow([metric, *outs.values()])


def main(experiment_list: list[str]):
    results = OrderedDict()
    for study_name in tqdm(experiment_list):
        study = optuna.load_study(study_name=study_name, storage=OPTUNA_DB)
        results[study_name] = get_trial_importances(study)
        write_importances_json(results[study_name], study_name)
        write_importances_csv(results[study_name], study_name)
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    experiment_list = [
        "SNN-SUPER-B-HERA-LATENCY-FC_LATENCY-100--False",
        "SNN-SUPER-B-HERA-LATENCY-FC_LATENCY-100--True",
        "SNN-SUPER-B-HERA-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-100--False",
        "SNN-SUPER-B-HERA-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-100--True-True",
        "SNN-SUPER-B-HERA-FORWARDSTEP-FC_FORWARD_STEP-100--direct-False",
        "SNN-SUPER-B-HERA-FORWARDSTEP-FC_FORWARD_STEP-100--direct-True",
        "SNN-SUPER-B-HERA-ANN-ANN-100-False",
        "SNN-SUPER-B-HERA-ANN-ANN-100-True",
        "SNN-SUPER-C-LOFAR-FORWARDSTEP-FC_FORWARD_STEP-15--direct-False",
        "SNN-SUPER-C-LOFAR-FORWARDSTEP-FC_FORWARD_STEP-15--direct-True",
        "SNN-SUPER-C-LOFAR-LATENCY-FC_LATENCY-15--False",
        "SNN-SUPER-C-LOFAR-LATENCY-FC_LATENCY-15--True",
        "SNN-SUPER-C-LOFAR-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-15--False",
        "SNN-SUPER-C-LOFAR-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-15--True",
        "SNN-SUPER-C-LOFAR-ANN-FC_ANN-15--False",
        "SNN-SUPER-C-LOFAR-ANN-FC_ANN-15--True",
    ]
    main(experiment_list)
