import json
import os
import numpy as np

E_AC = 9e-13


def open_spike_file(spike_file) -> np.ndarray:
    # Open spike_file with numpy
    return np.load(spike_file)

def open_dataset_file(dataset_file) -> dict:
    # Open dataset_file with json
    with open(dataset_file, "r") as f:
        return json.load(f)

def calculate_power_metrics(spike_recording, dataset_info) -> dict:
    # Calculate metrics
    total_spike_counts = np.sum(spike_recording, axis=0)
    num_neurons = []
    flops_per_layer = []
    for i in range(dataset_info["num_layers"] - 1):
        num_neurons.append(dataset_info["num_hidden"])
        if i == 0:
            flops_per_layer.append(dataset_info["num_inputs"] * dataset_info["num_hidden"] * dataset_info["exposure"])
        else:
            flops_per_layer.append(dataset_info["num_hidden"] * dataset_info["num_hidden"] * dataset_info["exposure"])
    num_neurons.append(dataset_info["num_outputs"])
    flops_per_layer.append(dataset_info["num_hidden"] * dataset_info["num_outputs"] * dataset_info["exposure"])
    flops_per_layer = np.asarray(flops_per_layer)
    spike_rates = total_spike_counts / num_neurons / dataset_info["num_samples"]
    energy_per_layer = np.array(flops_per_layer) * E_AC * spike_rates
    flops_per_layer_patch  = flops_per_layer * dataset_info["stride"]
    energy_per_layer_patch =  energy_per_layer * dataset_info["stride"]
    spike_rates_patch = spike_rates * dataset_info["stride"]
    flops_per_patch = np.sum(flops_per_layer_patch)
    energy_per_patch = np.sum(energy_per_layer_patch)
    return {"flops_per_layer": flops_per_layer.tolist(), "spike_rates": spike_rates.tolist(), "energy_per_layer": energy_per_layer.tolist(),
            "flops_per_layer_patch": flops_per_layer_patch.tolist(), "energy_per_layer_patch": energy_per_layer_patch.tolist(),
            "spike_rates_patch": spike_rates_patch.tolist(), "flops_per_patch": int(flops_per_patch), "energy_per_patch": float(energy_per_patch)}

def calculate_metrics(spike_file, dataset_file):
    # Open spike_file with numpy
    spike_recordings = open_spike_file(spike_file)
    # Open dataset_file with json
    dataset_info = open_dataset_file(dataset_file)
    # Calculate metrics
    power_metrics = calculate_power_metrics(spike_recordings, dataset_info)
    return power_metrics


def main():
    root_dir = "./lightning_logs/version_38/"
    spike_file = "full_spike_recording.npy"
    dataset_file = "dataset_info.json"
    metrics = calculate_metrics(os.path.join(root_dir, spike_file), os.path.join(root_dir, dataset_file))
    print(metrics)
    with open(os.path.join(root_dir, "power_metrics.json"), "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    main()