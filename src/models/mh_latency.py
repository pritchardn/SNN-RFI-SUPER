"""
This module contains the implementation of the LitFcDelta class, which is a PyTorch Lightning
module for a fully connected latency-coding model.
"""

import numpy as np
import snntorch.functional as SF

from interfaces.models.multihead_model import MHLitModel
from plotting import plot_example_inference, plot_example_mask


class MHLitLatency(MHLitModel):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        alpha: float,
        beta: float,
        head_width: int,
        head_stride: int,
        num_hidden_layers: int,
        learning_rate: float,
    ):
        super().__init__(
            num_inputs, num_hidden, num_outputs, alpha, beta, head_width, head_stride, num_hidden_layers, learning_rate
        )
        self.loss = SF.mse_temporal_loss(target_is_time=True)
        self.float()
        self.save_hyperparameters()

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
            plot_example_mask(
                np.moveaxis(y[0].detach().cpu().numpy(), 0, -1),
                str(self.current_epoch),
                self.trainer.log_dir,
            )

        self.log("val_loss", loss, sync_dist=True)
