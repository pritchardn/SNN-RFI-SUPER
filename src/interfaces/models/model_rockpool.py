import abc

import numpy as np
import lightning.pytorch as pl
import torch
from rockpool.nn.combinators import Sequential
from rockpool.nn.modules import LinearTorch, LIFTorch
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter

TAU_MEM = 0.05
TAU_SYN = 0.05


class BaseModelRockpool(pl.LightningModule):

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
        self.loss = torch.nn.MSELoss()

    def _init_layers(self):
        layers = []
        for i in range(self.num_layers):
            if i == 0:
                layers.append(LinearTorch((self.num_inputs, self.num_hidden), has_bias=False))
                layers.append(LIFTorch(  # TODO: Refinement of parameters
                    self.num_hidden,
                    tau_mem=TAU_MEM,
                    tau_syn=TAU_SYN,
                    threshold=1.0,
                    learning_window=0.2,
                    dt=0.001,
                ))
            elif i == self.num_layers - 1:
                layers.append(LinearTorch((self.num_hidden, self.num_outputs), has_bias=False))
                layers.append(LIFTorch(  # TODO: Refinement of parameters
                    self.num_outputs,
                    tau_mem=TAU_MEM,
                    tau_syn=TAU_SYN,
                    threshold=1.0,
                    learning_window=0.2,
                    dt=0.001,
                ))
            else:
                layers.append(LinearTorch((self.num_hidden, self.num_hidden), has_bias=False))
                layers.append(LIFTorch(  # TODO: Refinement of parameters
                    self.num_hidden,
                    tau_mem=TAU_MEM,
                    tau_syn=TAU_SYN,
                    threshold=1.0,
                    learning_window=0.2,
                    dt=0.001,
                ))
        self.model = Sequential(*layers)
        self.model = self.model.to("cuda")
        print(self.model)

    def set_converter(self, converter: SpikeConverter):
        self.converter = converter

    def calc_accuracy(self, y_hat, y):
        score = balanced_accuracy_score(y_hat.flatten(), y.flatten())
        self.log("accuracy", score)

    @abc.abstractmethod
    def forward(self, x):
        pass

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


class LitModelRockpool(BaseModelRockpool):

    def __init__(self,
                 num_inputs: int,
                 num_hidden: int,
                 num_outputs: int,
                 num_layers: int):
        super().__init__(num_inputs, num_hidden, num_outputs, num_layers)

    def forward(self, x):
        full_mem = []
        original_shape = x.shape
        data = torch.moveaxis(x, -1, 2)
        data = data.reshape(original_shape[0], -1, original_shape[-2])
        spike, mem, recording = self.model(data)
        spike = spike.reshape(original_shape[0], original_shape[1], original_shape[-2],
                              original_shape[-1])
        spike = spike.unsqueeze(2)
        spike = spike.moveaxis(-2, -1)
        return spike, full_mem


class LitModelPatchedRockpool(BaseModelRockpool):

    def __init__(self,
                 num_inputs: int,
                 num_hidden: int,
                 num_outputs: int,
                 num_layers: int):
        super().__init__(num_inputs, num_hidden, num_outputs, num_layers)

    def forward(self, x):
        full_mem = []
        self.model.reset_state()
        data = x.view(*(x.shape[:-2]), -1).squeeze(2)
        spike, _, _ = self.model(data)
        spike = spike.view(*(spike.shape[:-1]), -1, x.shape[-1])
        full_spike = spike.unsqueeze(2)
        return full_spike, full_mem
