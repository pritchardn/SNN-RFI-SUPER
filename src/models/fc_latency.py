import pytorch_lightning as pl
import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn


class LitFcLatency(pl.LightningModule):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        self.loss = SF.mse_temporal_loss(target_is_time=True)
        self.float()

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
        return torch.moveaxis(full_spike, 0, -1).unsqueeze(2), torch.moveaxis(full_mem, 0, -1).unsqueeze(2)

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
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.loss(spike_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
