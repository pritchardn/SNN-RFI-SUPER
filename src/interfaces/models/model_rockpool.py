import lightning.pytorch as pl
import torch
from rockpool.nn.combinators import Sequential
from rockpool.nn.modules import LinearTorch, LIFTorch
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter


class LitModelRockpool(pl.LightningModule):

    def __init__(self,
                 num_inputs: int,
                 num_hidden: int,
                 num_outputs: int,
                 num_layers: int):
        super().__init__()
        self.converter = None
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_layers = num_layers
        self._init_layers()

    def _init_layers(self):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(LinearTorch((self.num_inputs, self.num_hidden), has_bias=False))
            elif i == self.num_layers - 1:
                layers.append(LIFTorch(  # TODO: Refinement of parameters
                    (self.num_hidden, self.num_outputs),
                    tau_mem=0.002,
                    tau_syn=0.002,
                    threshold=1.0,
                    learning_window=0.2,
                    dt=0.001,
                ))
            else:
                layers.append(LIFTorch(  # TODO: Refinement of parameters
                    (self.num_hidden, self.num_hidden),
                    tau_mem=0.002,
                    tau_syn=0.002,
                    threshold=1.0,
                    learning_window=0.2,
                    dt=0.001,
                ))
        self.model = Sequential(*layers)

    def set_converter(self, converter: SpikeConverter):
        self.converter = converter

    def calc_accuracy(self, y_hat, y):
        score = balanced_accuracy_score(y_hat.flatten(), y.flatten())
        self.log("accuracy", score)

    def forward(self, x):
        full_spike = []
        full_mem = []
        self.model.reset_state()
        for t in range(x.shape[-1]):
            spike_out = []
            mem_out = []
            for step in range(x.shape[1]):  # [N x C x freq]
                data = x[:, step, 0, :, t]
                spike, mem, recording = self.model(data)
                spike_out.append(spike.squeeze())
                # mem_out.append(mem)
            full_spike.append(torch.stack(spike_out, dim=1))
            # full_mem.append(torch.stack(mem_out, dim=1))
        full_spike = torch.stack(full_spike, dim=0)  # [time x N x exp x C x freq]
        # full_mem = torch.stack(full_mem, dim=0)
        full_spike = torch.moveaxis(full_spike, 0, -1).unsqueeze(2)
        # full_mem = torch.moveaxis(full_mem, 0, -1).unsqueeze(2)
        return full_spike, full_mem

    def calc_loss(self, y_hat, y):
        loss = self.loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.calc_loss(spike_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.calc_loss(spike_hat, y)
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        # Convert output to true output
        output_pred = self.converter.decode_inference(spike_hat.detach().cpu().numpy())
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
