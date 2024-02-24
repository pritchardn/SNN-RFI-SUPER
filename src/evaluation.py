import json
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
    # Decode outputs into masks
    output = converter.decode_inference(full_spike_hat.detach().cpu().numpy())

    # Stitch masks together
    recon_output = reconstruct_patches(
        output, mask_orig.shape[-1], full_spike_hat.shape[-1]
    )
    # Calculate metrics on the whole dataset
    accuracy, mse, auroc, auprc, f1 = calculate_metrics(mask_orig, recon_output)
    output = json.dumps(
        {
            "accuracy": accuracy,
            "mse": mse,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
        }
    )
    # Write output
    with open(os.path.join(outdir, "metrics.json"), "w") as ofile:
        json.dump(output, ofile, indent=4)
    # Plot a sample
    for i in range(min(10, mask_orig.shape[0])):
        plot_final_examples(
            np.moveaxis(mask_orig[i], 0, -1),
            np.moveaxis(recon_output[i], 0, -1),
            f"final_{i}",
            outdir,
        )
    return accuracy, mse, auroc, auprc, f1


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = ensure_tflow(y_true)
    y_pred = ensure_tflow(y_pred)
    false_pos_rate, true_pos_rate, _ = roc_curve(
        y_true.flatten() > 0, y_pred.flatten() > 0
    )
    accuracy = accuracy_score(y_true.flatten(), y_pred.flatten())
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    auroc = auc(false_pos_rate, true_pos_rate)
    precision, recall, _ = precision_recall_curve(
        y_true.flatten() > 0, y_pred.flatten() > 0
    )
    auprc = auc(recall, precision)
    f1 = 2 * (precision * recall) / (precision + recall)
    return accuracy, mse, auroc, auprc, np.max(f1)
