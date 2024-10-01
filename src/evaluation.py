"""
Evaluation functions for the model
"""

import os

import numpy as np
import lightning.pytorch as pl
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


def final_evaluation(
    model: pl.LightningModule,
    data_module: ConfiguredDataModule,
    converter: SpikeConverter,
    data_orig,
    mask_orig,
    outdir: str,
):
    os.makedirs(outdir, exist_ok=True)
    # Run through the whole validation set
    full_spike_hat = []
    for x, y in tqdm(data_module.test_dataloader()):
        spike_hat, mem_hat = model(x)
        full_spike_hat.append(spike_hat)
    full_spike_hat = torch.cat(full_spike_hat, dim=1)
    # save full_spike_hat into .npy file
    np.save(
        os.path.join(outdir, "full_spike_hat.npy"),
        full_spike_hat.detach().cpu().numpy(),
    )
    # Decode outputs into masks
    output = converter.decode_inference(full_spike_hat.detach().cpu().numpy())
    # Stitch masks together
    recon_output = reconstruct_patches(
        output, mask_orig.shape[-1], full_spike_hat.shape[-1]
    )
    # Plot a sample
    for i in range(min(10, mask_orig.shape[0])):
        mask_example = mask_orig[i]
        mask_example[mask_example > 0.0] = 1.0
        plot_final_examples(
            np.moveaxis(data_orig[i], 0, -1),
            np.moveaxis(mask_example, 0, -1),
            np.moveaxis(recon_output[i], 0, -1),
            f"final_{i}",
            outdir,
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
