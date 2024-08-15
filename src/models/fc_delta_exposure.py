"""
This module contains the implementation of the LitDeltaExposure class, which is a PyTorch Lightning
module for a fully connected delta-encoding exposure model.
"""

import snntorch.functional as SF

from interfaces.models.model import LitModel


class LitFcDeltaExposure(LitModel):
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
        self.loss = SF.mse_temporal_loss(target_is_time=True)
        self.float()
        self.save_hyperparameters()

    def calc_loss(self, y_hat, y):
        loss = self.loss(y_hat, y)
        return loss
