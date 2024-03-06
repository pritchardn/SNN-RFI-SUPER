import torch
import torch.nn as nn

from data.utils import decode_delta_inference
from interfaces.models.model import LitModel


class LitFcDelta(LitModel):
    def __init__(
        self,
        num_inputs: int,
        num_hidden: int,
        num_outputs: int,
        beta: float,
        reconstruct_loss: bool,
        off_spikes: bool,
    ):
        super().__init__(num_inputs, num_hidden, num_outputs, beta, 2)
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

    def calc_loss(self, spike_hat, y):
        if self.reconstruct_loss:
            decoded_spike_hat = decode_delta_inference(spike_hat, use_numpy=False)
            decided_targets = decode_delta_inference(
                torch.moveaxis(y, 0, 1), use_numpy=False
            )
            return self.loss(decoded_spike_hat, decided_targets)
        else:
            return self.loss(spike_hat, torch.moveaxis(y, 0, 1))
