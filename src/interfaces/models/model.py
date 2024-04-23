"""
Base class for pytorch lightning models.
"""
import abc

import lightning.pytorch as pl
import torch
from torch import nn
import snntorch as snn
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter
from plotting import plot_example_inference


class BaseLitModel(pl.LightningModule):

    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
            beta: float,
            num_layers: int,
    ):
        super().__init__()
        self.converter = None
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.beta = beta
        self.num_layers = num_layers

        self.ann_layers = self._init_ann_layers()
        self.snn_layers = self._init_snn_layers()

    def _init_ann_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i == 0:
                layers.append(nn.Linear(self.num_inputs, self.num_hidden))
            elif i == self.num_layers - 1:
                layers.append(nn.Linear(self.num_hidden, self.num_outputs))
            else:
                layers.append(nn.Linear(self.num_hidden, self.num_hidden))
        return layers

    def _init_snn_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layers.append(snn.Leaky(beta=self.beta, learn_threshold=True))
        return layers

    def set_converter(self, converter: SpikeConverter):
        self.converter = converter

    def calc_accuracy(self, y_hat, y):
        score = balanced_accuracy_score(y_hat.flatten(), y.flatten())
        self.log("accuracy", score)

    def _init_membranes(self):
        return [lif.init_leaky() for lif in self.snn_layers]

    @abc.abstractmethod
    def forward(self, x):
        pass

    def calc_loss(self, y_hat, y):
        loss = self.loss(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _ = self(x)
        loss = self.calc_loss(spike_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _ = self(x)
        loss = self.calc_loss(spike_hat, y)
        if batch_idx == 0 and self.trainer.local_rank == 0:
            plot_example_inference(
                spike_hat[:, 0, 0, ::].detach().cpu(),
                str(self.current_epoch),
                self.trainer.log_dir,
            )
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _ = self(x)
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


class LitModel(BaseLitModel):

    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
            beta: float,
            num_layers: int,
    ):
        super().__init__(num_inputs, num_hidden, num_outputs, beta, num_layers)

    def _infer_slice(self, x, membranes):
        spike = None
        for n in range(self.num_layers):
            curr = self.ann_layers[n](x)
            spike, membranes[n] = self.snn_layers[n](curr, membranes[n])
            x = spike
        return spike, membranes[-1]

    def forward(self, x):
        full_spike = []
        full_mem = []
        # x -> [N x exp x C x freq x time]
        membranes = self._init_membranes()
        for t in range(x.shape[-1]):
            spike_out = []
            mem_out = []
            for step in range(x.shape[1]):  # [N x C x freq]
                data = x[:, step, 0, :, t]
                spike, mem = self._infer_slice(data, membranes)
                spike_out.append(spike)
                mem_out.append(mem)
            full_spike.append(torch.stack(spike_out, dim=1))
            full_mem.append(torch.stack(mem_out, dim=1))
        full_spike = torch.stack(full_spike, dim=0)  # [time x N x exp x C x freq]
        full_mem = torch.stack(full_mem, dim=0)
        full_spike = torch.moveaxis(full_spike, 0, -1).unsqueeze(2)
        full_mem = torch.moveaxis(full_mem, 0, -1).unsqueeze(2)
        print(full_spike.shape)
        return torch.moveaxis(full_spike, 0, 1), torch.moveaxis(full_mem, 0, 1)


class LitPatchedModel(BaseLitModel):

    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
            beta: float,
            num_layers: int,
    ):
        super().__init__(num_inputs, num_hidden, num_outputs, beta, num_layers)

    def _infer_patch(self, x, membranes):
        spike = None
        for n in range(self.num_layers):
            curr = self.ann_layers[n](x)
            spike, membranes[n] = self.snn_layers[n](curr, membranes[n])
            x = spike
        return spike, membranes[-1]

    def forward(self, x):
        full_spike = []
        full_mem = []
        # x -> [N x exp x C x freq x time]
        membranes = self._init_membranes()
        spike_out = []
        mem_out = []
        for step in range(x.shape[1]):  # [N x C x freq]
            data = x[:, step, 0, :, :]
            data = data.view(*(data.shape[:-2]), -1)
            spike, mem = self._infer_patch(data, membranes)
            spike = spike.view(*(spike.shape[:-1]), -1, x.shape[-1])
            mem = mem.view(*(mem.shape[:-1]), -1, x.shape[-1])
            spike_out.append(spike)
            mem_out.append(mem)
        full_spike.append(torch.stack(spike_out, dim=1))
        full_mem.append(torch.stack(mem_out, dim=1))
        full_spike = torch.stack(full_spike, dim=0)  # [time x N x exp x C x freq]
        full_mem = torch.stack(full_mem, dim=0)
        full_spike = full_spike.squeeze(0).unsqueeze(2)
        full_mem = full_mem.squeeze(0).unsqueeze(2)
        return torch.moveaxis(full_spike, 0, 1), torch.moveaxis(full_mem, 0, 1)
