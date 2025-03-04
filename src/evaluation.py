"""
Evaluation functions for the model
"""
import json
import os

import lightning.pytorch as pl
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    mean_squared_error,
    auc,
    precision_recall_curve,
)
from tqdm import tqdm

from data.data_module import ConfiguredDataModule
from data.utils import reconstruct_patches, ensure_tflow
from interfaces.data.spiking_data_module import SpikeConverter
from plotting import plot_final_examples


def plot_example_raster(
        spike_x, frequency_width, stride, exposure, i, title: str, mode=1, outdir="./"
):
    # plt.tight_layout()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc("axes", labelsize=10 * mode)
    plt.rc("xtick", labelsize=8 * mode)
    plt.rc("ytick", labelsize=8 * mode)
    plt.figure(figsize=(10, 5))
    example = spike_x
    num_polarizations = example.shape[1]
    if num_polarizations == 1:
        example = example.squeeze(1)  # Remove channel dimension
    else:
        example = example[:, 0, ...]
    out = np.zeros((frequency_width, stride * exposure))
    for t in range(example.shape[-1]):  # t
        out[:, t * exposure: (t + 1) * exposure] = np.moveaxis(example[:, :, t], 0, -1)
    if min(spike_x.flatten()) < 0:
        ticks = [-1, 0, 1]
        cmap = plt.get_cmap("viridis", 3)
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist = [cmap(1), cmap(0), cmap(2)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, cmap.N
        )
    else:
        ticks = [0, 1]
        cmap = plt.get_cmap("viridis", 3)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # cmaplist[0] = (1.0, 1.0, 1.0, 1.0)
        cmaplist = [cmap(0), cmap(2)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, cmap.N - 1
        )
    plt.imshow(out, cmap=cmap)
    plt.gca().invert_yaxis()
    plt.ylabel("Frequency bin")
    plt.xlabel("Time [s]")

    plt.colorbar(location="right", ticks=ticks, shrink=0.5 * mode)
    plt.savefig(os.path.join(outdir, f"raster_{title}_example_{i}.png"), bbox_inches="tight", dpi=300)
    plt.close()


def final_evaluation(
        model: pl.LightningModule,
        data_module: ConfiguredDataModule,
        converter: SpikeConverter,
        data_orig,
        mask_orig,
        test_patches_x,
        test_patches_y,
        exposure,
        outdir: str,
):
    os.makedirs(outdir, exist_ok=True)
    # Run through the whole validation set
    full_spike_hat = []
    spike_recordings = []
    for x, y in tqdm(data_module.test_dataloader()):
        spike_hat, spike_recording = model(x)
        full_spike_hat.append(spike_hat)
        spike_recordings.append(spike_recording)
    full_spike_hat = torch.cat(full_spike_hat, dim=1)
    full_spike_recordings = np.concat(spike_recordings)
    # save full_spike_hat into .npy file
    np.save(
        os.path.join(outdir, "full_spike_hat.npy"),
        full_spike_hat.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(outdir, "full_spike_recording.npy"),
        full_spike_recordings,
    )
    # Dataset information
    dataset_info = {}
    dataset_info["num_batches"] = len(data_module.test_dataloader())
    dataset_info["batch_sizes"] = data_module.batch_size
    dataset_info["exposure"] = exposure
    dataset_info["num_samples"] = len(data_module.test_dataloader().dataset)
    dataset_info["num_outputs"] = model.num_outputs
    dataset_info["num_hidden"] = model.num_hidden
    dataset_info["num_layers"] = model.num_layers
    dataset_info["num_inputs"] = model.num_inputs
    # I need the stride of the input
    dataset_info["stride"] = data_module.stride
    with open(os.path.join(outdir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f)
    # Decode outputs into masks
    inference = full_spike_hat.detach().cpu().numpy()
    output = converter.decode_inference(inference)
    # Stitch masks together
    recon_output = reconstruct_patches(
        output, mask_orig.shape[-1], full_spike_hat.shape[-1]
    )
    inference = np.moveaxis(inference, 0, 1)
    # Plot a sample
    print("PLOTTING")
    for i in tqdm(range(min(10, mask_orig.shape[0]))):
        mask_example = mask_orig[i]
        mask_example[mask_example > 0.0] = 1.0
        plot_final_examples(
            np.moveaxis(data_orig[i], 0, -1),
            np.moveaxis(mask_example, 0, -1),
            np.moveaxis(recon_output[i], 0, -1),
            f"final_{i}",
            outdir,
        )
    for i in tqdm(range(min(10, output.shape[0]))):
        plot_final_examples(
            np.moveaxis(test_patches_x[i], 0, -1),
            np.moveaxis(test_patches_y[i], 0, -1),
            np.moveaxis(output[i], 0, -1),
            f"final_patch_{i}",
            outdir
        )
        plot_example_raster(
            inference[i],
            32,
            32,
            exposure,
            i,
            f"final_patch",
            mode=1,
            outdir=outdir
        )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = ensure_tflow(y_true)
    y_pred = ensure_tflow(y_pred)
    if np.any(y_true):
        pos_label = 1
    else:
        pos_label = 0
    false_pos_rate, true_pos_rate, _ = roc_curve(
        y_true.flatten() > 0, y_pred.flatten() > 0, pos_label=pos_label
    )
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten() > 0)
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten() > 0)
    auroc = auc(false_pos_rate, true_pos_rate)
    precision, recall, _ = precision_recall_curve(
        y_true.flatten() > 0, y_pred.flatten() > 0, pos_label=pos_label
    )
    auprc = auc(recall, precision)
    f1 = 2 * (precision * recall) / (precision + recall)
    f1 = np.nan_to_num(f1, nan=0.0)
    auroc = np.nan_to_num(auroc, nan=0.0)
    return accuracy, mse, auroc, auprc, np.max(f1)
