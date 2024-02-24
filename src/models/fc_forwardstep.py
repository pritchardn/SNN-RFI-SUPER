import snntorch as snn
import snntorch.functional as SF
import torch.nn as nn

from interfaces.models.model import LitModel


class LitFcForwardStep(LitModel):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, beta: float):
        super().__init__()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, learn_threshold=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, learn_threshold=True)
        self.loss = SF.mse_temporal_loss(target_is_time=True)
        self.float()
        self.save_hyperparameters()

    def calc_loss(self, y_hat, y):
        loss = self.loss(y_hat, y)
        return loss
