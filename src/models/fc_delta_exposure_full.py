"""
This module contains the implementation of the LitDeltaExposureFull class, which is a PyTorch Lightning
module for a fully connected delta-encoding exposure model with full output encoding.
"""

import torch
from torch import nn

from interfaces.models.model import LitModel


class LitFcDeltaExposureFull(LitModel):
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int,
            beta: float,
            num_layers: int,
            recurrent: bool = False,
    ):
        super().__init__(
            num_inputs, num_hidden, num_outputs, beta, num_layers, recurrent
        )
        self.loss = nn.MSELoss()
        self.float()
        self.save_hyperparameters()

    def calc_loss(self, y_hat, y):
        y = torch.moveaxis(y, 0, 1)
        loss = self.loss(y_hat, y)
        return loss
