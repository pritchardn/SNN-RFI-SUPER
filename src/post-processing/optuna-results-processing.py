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
        "SNN-SUPER-HERA-ANN-100-128",
        "SNN-SUPER-HERA-DELTA-100-128",
        "SNN-SUPER-HERA-LATENCY-100",
        "SNN-SUPER-HERA-RATE-100-128",
        "SNN-SUPER-HERA-FORWARDSTEP-100-128-direct",
        "SNN-SUPER-HERA-FORWARDSTEP-100-128-first",
        "SNN-SUPER-HERA-FORWARDSTEP-100-128-latency",
    ]
    for experiment_name in experiment_list:
        main(experiment_name + "_best_trial.json")
