import numpy as np
import snntorch as snn
import snntorch.functional as SF
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from interfaces.models.model import LitModel
from plotting import plot_example_inference, plot_example_mask


class LitFcForwardStep(LitModel):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, learn_threshold=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, learn_threshold=True)
        self.loss = SF.mse_temporal_loss()  # TODO: Loss function incorrect.
        self.float()
        self.save_hyperparameters()

    def calc_loss(self, y_hat, y):
        loss = self.loss(y_hat, y)
        return loss
