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
    with open(f"{filename[:-5]}.csv", "w") as ofile:
        fieldnames = collapsed_trials[0].keys()
        writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        writer.writeheader()
        for trial in collapsed_trials:
            writer.writerow(trial)


if __name__ == "__main__":
    main("SNN-SUPER-HERA-RATE-10_best_trial.json")
