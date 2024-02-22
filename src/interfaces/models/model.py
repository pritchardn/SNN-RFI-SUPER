from abc import abstractmethod

import pytorch_lightning as pl
import torch
from sklearn.metrics import balanced_accuracy_score
from torch.optim.lr_scheduler import ReduceLROnPlateau

from plotting import plot_example_inference


class LitModel(pl.LightningModule):
    def calc_accuracy(self, y_hat, y):
        score = balanced_accuracy_score(y_hat.flatten(), y.flatten())
        self.log("accuracy", score)

    def forward(self, x):
        full_spike = []
        full_mem = []
        # x -> [N x exp x C x freq x time]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for t in range(x.shape[-1]):
            spike_out = []
            mem_out = []
            for step in range(x.shape[1]):  # [N x C x freq]
                data = x[:, step, 0, :, t]
                cur1 = self.fc1(data)
                spike1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spike1)
                spike2, mem2 = self.lif2(cur2, mem2)
                spike_out.append(spike2)
                mem_out.append(mem2)
            full_spike.append(torch.stack(spike_out, dim=1))
            full_mem.append(torch.stack(mem_out, dim=1))
        full_spike = torch.stack(full_spike, dim=0)  # [time x N x exp x C x freq]
        full_mem = torch.stack(full_mem, dim=0)
        full_spike = torch.moveaxis(full_spike, 0, -1).unsqueeze(2)
        full_mem = torch.moveaxis(full_mem, 0, -1).unsqueeze(2)
        return torch.moveaxis(full_spike, 0, 1), torch.moveaxis(full_mem, 0, 1)

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
        if batch_idx == 0 and self.trainer.local_rank == 0:
            plot_example_inference(
                spike_hat[:, 0, 0, ::].detach().cpu(),
                str(self.current_epoch),
                self.trainer.log_dir,
            )
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.calc_loss(spike_hat, y)
        self.log("test_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
