"""
This script is used to convert the best trial json files to csv files.
"""

import csv
import json


def _collapse_params_values(trial: dict) -> dict:
    params = trial["params"]
    values = trial["values"]
    collapsed = {"id": trial["id"]}
    collapsed.update(values)
    collapsed.update(params)
    return collapsed


def main(filename: str):
    with open(filename, "r") as file:
        data = json.load(file)
    print(json.dumps(data, indent=4))
    collapsed_trials = []
    for trial in data:
        collapsed_trials.append(_collapse_params_values(trial))
    with open(f"{filename[:-5]}.csv", "w", newline="") as ofile:
        fieldnames = collapsed_trials[0].keys()
        writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        writer.writeheader()
        for trial in collapsed_trials:
            writer.writerow(trial)


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
        "SNN-SUPER-B-LOFAR-LATENCY-FC_LATENCY-50--False",
        "SNN-SUPER-B-LOFAR-LATENCY-FC_LATENCY-50--True",
        "SNN-SUPER-B-LOFAR-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-50--False",
        "SNN-SUPER-B-LOFAR-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-50--True",
        "SNN-SUPER-B-LOFAR-FORWARDSTEP-FC_FORWARD_STEP-50--direct-False",
        "SNN-SUPER-B-LOFAR-FORWARDSTEP-FC_FORWARD_STEP-50--direct-True",
        "SNN-SUPER-B-LOFAR-ANN-FC_ANN-50--False",
        "SNN-SUPER-B-LOFAR-ANN-FC_ANN-50--True",
    ]
    for experiment_name in experiment_list:
        main(experiment_name + "_best_trial.json")
