import pytorch_lightning as pl
import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from plotting import plot_example_inference, plot_example_mask


class LitFcLatency(pl.LightningModule):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, learn_threshold=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, learn_threshold=True)
        self.loss = SF.mse_temporal_loss(target_is_time=True)
        self.float()

    def calc_accuracy(self, y_hat, y):
        score = balanced_accuracy_score(y_hat.flatten(), y.flatten())
        self.log("accuracy", score)

    def forward(self, x):
        full_spike = []
        full_mem = []
        # x -> [N x exp x C x freq x time]
        for t in range(x.shape[-1]):
            mem1 = self.lif1.init_leaky()
            mem2 = self.lif2.init_leaky()
            spike_out = []
            mem_out = []
            for step in range(x.shape[1]):
                data = x[:, step, 0, :, t]  # [N x C x freq]
                cur1 = self.fc1(data)
                spike1, mem1 = self.lif1(cur1, mem1)
                cur2 = self.fc2(spike1)
                spike2, mem2 = self.lif2(cur2, mem2)
                spike_out.append(spike2)
                mem_out.append(mem2)
            # All examples at time t in original spectrogram
            full_spike.append(torch.stack(spike_out, dim=1))
            full_mem.append(torch.stack(mem_out, dim=1))
        full_spike = torch.stack(full_spike, dim=0)  # [time x N x exp x C x freq]
        full_mem = torch.stack(full_mem, dim=0)
        full_spike = torch.moveaxis(full_spike, 0, -1).unsqueeze(2)
        full_mem = torch.moveaxis(full_mem, 0, -1).unsqueeze(2)
        return torch.moveaxis(full_spike, 0, 1), torch.moveaxis(full_mem, 0, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.loss(spike_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.loss(spike_hat, y)
        if batch_idx == 0:
            plot_example_inference(
                spike_hat[:, 0, 0, ::].detach().cpu(), str(self.current_epoch), self.trainer.log_dir
            )
            plot_example_mask(np.moveaxis(y[0].detach().cpu().numpy(), 0, -1), str(self.current_epoch), self.trainer.log_dir)

        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.loss(spike_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
