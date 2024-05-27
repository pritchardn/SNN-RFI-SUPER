"""
Fully connected artificial neural network model for spectrograms.
"""

import lightning.pytorch as pl
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter


class LitFcANN(pl.LightningModule):
    def __init__(
        self, num_inputs: int, num_hidden: int, num_outputs: int, num_layers: int
    ):
        super().__init__()
        self.converter = None
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self.layers = self._init_layers()
        self.loss = nn.MSELoss()

    def _init_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.num_inputs, self.num_hidden))
                layers.append(nn.ReLU())
            elif i == self.num_layers - 1:
                layers.append(nn.Linear(self.num_hidden, self.num_outputs))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(self.num_hidden, self.num_hidden))
        return layers

    def set_converter(self, converter: SpikeConverter):
        self.converter = converter

    def calc_accuracy(self, y_hat, y):
        score = balanced_accuracy_score(y_hat.flatten(), y.flatten())
        self.log("accuracy", score)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x, None

    def calc_loss(self, y_hat, y):
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.calc_loss(y_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        loss = self.calc_loss(y_hat, y)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output_pred, _ = self(x)
        output_pred = output_pred.detach().cpu().numpy()
        # Convert output to true output
        accuracy, mse, auroc, auprc, f1 = calculate_metrics(
            y.detach().cpu().numpy(), output_pred
        )
        self.log("test_accuracy", accuracy, sync_dist=True)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_auroc", auroc, sync_dist=True)
        self.log("test_auprc", auprc, sync_dist=True)
        self.log("test_f1", f1, sync_dist=True)
        return accuracy, mse, auroc, auprc, f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
