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
    return accuracy, mse, auroc, auprc, f1


def final_evaluation(
    model: pl.LightningModule,
    data_module: ConfiguredDataModule,
    converter: SpikeConverter,
    mask_orig,
):
    # Run through the whole validation set
    full_spike_hat = []
    for x, y in tqdm(data_module.test_dataloader()):
        spike_hat, mem_hat = model(x)
        full_spike_hat.append(spike_hat)
    full_spike_hat = torch.cat(full_spike_hat, dim=0)
    # Decode outputs into masks
    output = converter.decode_inference(full_spike_hat.detach().cpu().numpy())

    # Stitch masks together
    recon_output = reconstruct_patches(output, mask_orig.shape[-1], 32)
    # Calculate metrics on the whole dataset
    accuracy, mse, auroc, auprc, f1 = _calculate_metrics(mask_orig, recon_output)
    model.log("accuracy", accuracy)
    model.log("mse", mse)
    model.log("auroc", auroc)
    model.log("auprc", auprc)
    model.log("f1", f1)
    # Plot a sample


def main():
    EXPOSURE = 16
    TAU = 1.0
    BETA = 0.95
    data_builder = DataModuleBuilder()
    data_source = HeraDataLoader("./data", patch_size=32, stride=32)
    data_builder.set_dataset(data_source)
    spike_converter = LatencySpikeConverter(exposure=EXPOSURE, tau=TAU, normalize=True)
    data_builder.set_encoding(spike_converter)
    data_module = data_builder.build(32)
    print("Built data module")
    model = LitFcLatency(32, 128, 32, BETA)
    print("Built model")
    trainer = pl.trainer.Trainer(max_epochs=1, benchmark=True)
    # trainer.fit(model, data_module)
    model.eval()
    mask_orig = reconstruct_patches(data_source.fetch_train_y(), 512, 32)
    final_evaluation(model, data_module, spike_converter, mask_orig)


if __name__ == "__main__":
    main()
