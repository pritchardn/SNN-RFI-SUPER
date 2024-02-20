import snntorch as snn
import torch
import torch.nn as nn

from data.utils import decode_delta_inference
from interfaces.models.model import LitModel


class LitFcDelta(LitModel):
    def __init__(self, num_inputs: int, num_hidden: int, num_outputs: int, beta: float,
                 reconstruct_loss: bool):
        super().__init__()
        if num_outputs != num_inputs * 2:
            raise ValueError("num_outputs must be 2 * num_inputs for delta-coding")
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, learn_threshold=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, learn_threshold=True)
        self.loss = nn.HuberLoss()
        self.float()
        self.save_hyperparameters()
        self.reconstruct_loss = reconstruct_loss

    def forward(self, x):
        full_spike = []
        full_mem = []
        # x -> [N x exp x C x freq x time]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        for t in range(x.shape[-1]):
            spike_out = []
            mem_out = []
            for step in range(x.shape[1]):  # Should always be 1
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

    def calc_loss(self, spike_hat, y):
        if self.reconstruct_loss:
            decoded_spike_hat = decode_delta_inference(spike_hat, use_numpy=False)
            decided_targets = decode_delta_inference(torch.moveaxis(y, 0, 1), use_numpy=False)
            return self.loss(decoded_spike_hat, decided_targets)
        else:
            return self.loss(spike_hat, torch.moveaxis(y, 0, 1))
