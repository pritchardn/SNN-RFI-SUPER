"""
This module contains the implementation of the LitFcDelta class, which is a PyTorch Lightning
module for a fully connected rate-coding model. Operating in patched inference mode.
"""

import torch

from interfaces.models.model import LitPatchedModel
from loss_functions.mse_count_balanced_loss import mse_count_loss_balanced


class LitFcPRate(LitPatchedModel):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        beta: float,
        num_layers: int,
    ):
        super().__init__(num_inputs, num_hidden, num_outputs, beta, num_layers)
        self.loss = mse_count_loss_balanced(correct_rate=0.8, incorrect_rate=0.2)
        self.float()
        self.save_hyperparameters()

    def calc_loss(self, y_hat, y):
        loss = 0.0
        for i in range(y.shape[0]):
            example_loss = 0.0
            for t in range(y.shape[1]):
                tslice = y[i, t, ::]
                targets = tslice[torch.where(tslice >= 0)[0]]
                spikes = y_hat[:, i, :, t]
                example_loss += self.loss(spikes, targets)
            example_loss /= y.shape[1]
            loss += example_loss
        return loss
