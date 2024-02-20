import csv
import json
import os

import numpy as np


def process_metric_file(filename: str):
    with open(filename, "r") as file:
        data = json.load(file)
    data = json.loads(data)
    return data


def calculate_results_summary(results: list):
    metrics = list(set(key for result in results for key in result.keys()))
    summary = {}
    for metric in metrics:
        values = [result[metric] for result in results]  # Want it to error if not found
        summary[metric] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
        }
    summary["n"] = len(results)
    return summary


def write_output_csv(filename: str, data: list):
    with open(filename, "w") as ofile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def write_output_json(filename: str, data: dict):
    with open(filename, "w") as ofile:
        json.dump(data, ofile, indent=4)


def main(in_dirname: str, out_dirname: str, out_filename: str = "summary"):
    # Collect all metrics.json files in directory and subdirectories
    metrics_files = []
    results = []
    for root, dirs, files in os.walk(in_dirname):
        for file in files:
            if file == "metrics.json":
                metrics_files.append(os.path.join(root, file))
                results.append(process_metric_file(os.path.join(root, file)))
    print(len(metrics_files))
    print(json.dumps(results, indent=4))
    summary = calculate_results_summary(results)
    print(json.dumps(summary, indent=4))
    write_output_csv(os.path.join(out_dirname, f"{out_filename}.csv"), results)
    write_output_json(os.path.join(out_dirname, f"{out_filename}.json"), summary)


if __name__ == "__main__":
    root_dir = "outputs/snn-super/FC/LATENCY/HERA/128/1.0/"
    log_dir = "lightning_logs/"
    main(os.path.join(root_dir, log_dir), root_dir)
