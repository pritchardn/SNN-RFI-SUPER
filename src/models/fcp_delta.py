"""
This module contains the implementation of the LitFcDelta class, which is a PyTorch Lightning
module for a fully connected delta-coding model. Operating in patched inference mode.
"""

import torch
from torch import nn

from data.utils import decode_delta_inference
from interfaces.models.model import LitPatchedModel


class LitFcPDelta(LitPatchedModel):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        beta: float,
        reconstruct_loss: bool,
        off_spikes: bool,
        num_layers: int,
    ):
        super().__init__(num_inputs, num_hidden, num_outputs, beta, num_layers)
        if off_spikes:
            if num_outputs != num_inputs * 2:
                raise ValueError("num_outputs must be 2 * num_inputs for delta-coding")
        else:
            if num_outputs != num_inputs:
                raise ValueError("num_outputs must be num_inputs for delta-coding")
        self.loss = nn.HuberLoss()
        self.float()
        self.save_hyperparameters()
        self.reconstruct_loss = reconstruct_loss

    def calc_loss(self, y_hat, y):
        if self.reconstruct_loss:
            decoded_spike_hat = decode_delta_inference(y_hat, use_numpy=False)
            decided_targets = decode_delta_inference(
                torch.moveaxis(y, 0, 1), use_numpy=False
            )
            return self.loss(decoded_spike_hat, decided_targets)
        return self.loss(y_hat, torch.moveaxis(y, 0, 1))
