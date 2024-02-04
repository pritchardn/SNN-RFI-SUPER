import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import (
    roc_curve,
    accuracy_score,
    mean_squared_error,
    auc,
    precision_recall_curve,
)
from tqdm import tqdm
import os
import json

from data.data_loaders import HeraDataLoader
from data.data_module import ConfiguredDataModule
from data.data_module_builder import DataModuleBuilder
from data.spike_converters import LatencySpikeConverter
from data.utils import reconstruct_patches, ensure_tflow
from interfaces.data.spiking_data_module import SpikeConverter
from models.fc_latency import LitFcLatency


def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
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


def final_evaluation(
        model: pl.LightningModule,
        data_module: ConfiguredDataModule,
        converter: SpikeConverter,
        mask_orig,
        outdir: str
):
    # Run through the whole validation set
    full_spike_hat = []
    for x, y in tqdm(data_module.test_dataloader()):
        spike_hat, mem_hat = model(x)
        full_spike_hat.append(spike_hat)
    full_spike_hat = torch.cat(full_spike_hat, dim=1)
    # Decode outputs into masks
    output = converter.decode_inference(full_spike_hat.detach().cpu().numpy())

    # Stitch masks together
    recon_output = reconstruct_patches(output, mask_orig.shape[-1], full_spike_hat.shape[-1])
    # Calculate metrics on the whole dataset
    accuracy, mse, auroc, auprc, f1 = _calculate_metrics(mask_orig, recon_output)
    output = json.dumps(
        {
            "accuracy": accuracy,
            "mse": mse,
            "auroc": auroc,
            "auprc": auprc,
            "f1": f1,
        })
    # Write output
    with open(os.path.join(outdir, "metrics.json"), "w") as ofile:
        json.dump(output, ofile, indent=4)
    # Plot a sample


def main():
    EXPOSURE = 8
    TAU = 1.0
    BETA = 0.95
    STRIDE = 32
    ORIGINAL_SHAPE = (512, 512)
    data_builder = DataModuleBuilder()
    data_source = HeraDataLoader("./data", patch_size=STRIDE, stride=STRIDE, limit=0.1)
    data_builder.set_dataset(data_source)
    spike_converter = LatencySpikeConverter(exposure=EXPOSURE, tau=TAU, normalize=True)
    data_builder.set_encoding(spike_converter)
    data_module = data_builder.build(32)
    print("Built data module")
    model = LitFcLatency(32, 128, 32, BETA)
    print("Built model")
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=5, min_delta=1e-4)
    trainer = pl.trainer.Trainer(max_epochs=50, benchmark=True, callbacks=[early_stopping_callback])
    trainer.fit(model, data_module)
    model.eval()
    mask_orig = reconstruct_patches(data_source.fetch_test_y(), ORIGINAL_SHAPE[0], STRIDE)
    final_evaluation(model, data_module, spike_converter, mask_orig, trainer.log_dir)


if __name__ == "__main__":
    main()
