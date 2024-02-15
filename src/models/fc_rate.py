import pytorch_lightning as pl
import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from plotting import plot_example_inference


class LitFcRate(pl.LightningModule):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, learn_threshold=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, learn_threshold=True)
        self.pos_loss = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
        self.neg_loss = SF.mse_count_loss(correct_rate=0.2, incorrect_rate=0.8)
        self.float()
        self.save_hyperparameters()

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
            for step in range(x.shape[1]):
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

    def _calc_loss(self, spike_hat, y):
        loss = 0.0
        for i in range(y.shape[0]):
            example_loss = 0.0
            for t in range(y.shape[1]):
                tslice = y[i, t, ::]
                targets = tslice[torch.where(tslice >= 0)[0]]
                spikes = spike_hat[:, i, :, t]
                if len(targets) == 0:
                    continue
                    targets = torch.arange(0, spikes.shape[-1], device=self.device)
                    example_loss += self.neg_loss(spikes, targets)
                else:
                    example_loss += self.pos_loss(spikes, targets)
            example_loss /= y.shape[1]
            loss += example_loss
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self._calc_loss(spike_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self._calc_loss(spike_hat, y)
        if batch_idx == 0:
            plot_example_inference(
                spike_hat[:, 0, 0, ::].detach().cpu(),
                str(self.current_epoch),
                self.trainer.log_dir,
            )
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self._calc_loss(spike_hat, y)
        self.log("test_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
