"""
This script processes the results of the trials and calculates the summary
statistics for the metrics.
"""

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
            "mean": np.nanmean(values),
            "std": np.nanstd(values),
            "min": np.nanmin(values),
            "max": np.nanmax(values),
        }
    summary["n"] = len(results)
    return summary


def write_output_csv(filename: str, data: list):
    with open(filename, "w", newline="") as ofile:
        fieldnames = data[0].keys()
        writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)


def write_output_json(filename: str, data: dict):
    with open(filename, "w") as ofile:
        json.dump(data, ofile, indent=4)


def write_summary_csv(filename: str, data: dict):
    # Convert dict into list
    data_list = []
    num_trials = 0
    for key, value in data.items():
        if isinstance(value, int):
            num_trials = value
            continue
        value["title"] = key
        data_list.append(value)
    for row in data_list:
        row["n"] = num_trials
    print(data_list)
    # Write output
    with open(filename, "w", newline="") as ofile:
        fieldnames = data_list[0].keys()
        writer = csv.DictWriter(ofile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_list:
            writer.writerow(row)


def main(
    in_dirname: str, out_dirname: str, out_filename: str = "summary", limit: int = None
):
    # Collect all metrics.json files in directory and subdirectories
    metrics_files = []
    results = []
    count = 0
    for root, dirs, files in os.walk(in_dirname):
        for file in files:
            if file == "metrics.json":
                metrics_files.append(os.path.join(root, file))
                results.append(process_metric_file(os.path.join(root, file)))
                count += 1
        if limit is not None and count >= limit:
            break
    print(len(metrics_files))
    print(json.dumps(results, indent=4))
    summary = calculate_results_summary(results)
    print(json.dumps(summary, indent=4))
    write_output_csv(os.path.join(out_dirname, f"{out_filename}.csv"), results)
    write_output_json(os.path.join(out_dirname, f"{out_filename}.json"), summary)
    write_summary_csv(os.path.join(out_dirname, f"{out_filename}_summary.csv"), summary)


def main_process_supercomputer():
    for model, encoding in [
        ("FC_DELTA", "DELTA"),
        ("FC_LATENCY", "LATENCY"),
        ("FC_RATE", "RATE"),
        ("FC_FORWARD_STEP", ("FORWARDSTEP", "first")),
        ("FC_FORWARD_STEP", ("FORWARDSTEP", "direct")),
        ("FC_FORWARD_STEP", ("FORWARDSTEP", "latency")),
    ]:
        if model == "FC_FORWARD_STEP":
            encoding, exposure_mode = encoding
            root_dir = f".{os.sep}outputs{os.sep}snn-super{os.sep}{model}{os.sep}{encoding}{os.sep}HERA{os.sep}2{os.sep}128{os.sep}1.0{os.sep}{exposure_mode}"
            print(root_dir)
            log_dir = f"lightning_logs{os.sep}"
            output_dir = f".{os.sep}"
            main(
                os.path.join(root_dir, log_dir),
                output_dir,
                f"{model}-{exposure_mode}",
                limit=50,
            )
        else:
            root_dir = f".{os.sep}outputs{os.sep}snn-super{os.sep}{model}{os.sep}{encoding}{os.sep}HERA{os.sep}2{os.sep}128{os.sep}1.0"
            log_dir = f"lightning_logs{os.sep}"
            output_dir = f".{os.sep}"
            main(os.path.join(root_dir, log_dir), output_dir, model, limit=50)


def main_process_custom():
    root_dir = (
        "./snn-super/FC_LATENCY_XYLO/LATENCY/HERA_POLAR_FULL/True/1.0/"
    )
    log_dir = "lightning_logs/"
    output_dir = "./"
    main(os.path.join(root_dir, log_dir), output_dir, "LATENCY_XYLO_HERA_DIVNORM_POLAR_FULL")


if __name__ == "__main__":
    main_process_custom()
