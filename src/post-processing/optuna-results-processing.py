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
        "SNN-SUPER-C-LOFAR-FORWARDSTEP-FC_FORWARD_STEP-15--direct-True",
        "SNN-SUPER-C-LOFAR-FORWARDSTEP-FC_FORWARD_STEP-15--direct-False",
        "SNN-SUPER-C-LOFAR-LATENCY-FC_LATENCY-15--True",
        "SNN-SUPER-C-LOFAR-LATENCY-FC_LATENCY-15--False",
        "SNN-SUPER-C-LOFAR-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-15--True",
        "SNN-SUPER-C-LOFAR-DELTA_EXPOSURE-FC_DELTA_EXPOSURE-15--False",
        "SNN-SUPER-C-LOFAR-ANN-FC_ANN-15--True",
        "SNN-SUPER-C-LOFAR-ANN-FC_ANN-15--False",
    ]
    for experiment_name in experiment_list:
        main(experiment_name + "_best_trial.json")
