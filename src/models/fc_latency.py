import numpy as np
import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from interfaces.models.model import LitModel
from plotting import plot_example_inference, plot_example_mask


class LitFcLatency(LitModel):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, learn_threshold=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, learn_threshold=True)
        self.loss = SF.mse_temporal_loss(target_is_time=True)
        self.float()
        self.save_hyperparameters()

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
            plot_example_mask(
                np.moveaxis(y[0].detach().cpu().numpy(), 0, -1),
                str(self.current_epoch),
                self.trainer.log_dir,
            )

        self.log("val_loss", loss, sync_dist=True)
