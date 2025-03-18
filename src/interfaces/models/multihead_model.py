"""
Base class for fullwidth pytorch lightning models.
"""


"""
Base class for pytorch lightning models. Handles both fully connected Lif and Recurrent Models.
"""

import abc

import lightning.pytorch as pl
import torch
from torch import nn
import snntorch as snn
import snntorch.functional as functional
from sklearn.metrics import balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from torch.optim.lr_scheduler import ReduceLROnPlateau

from evaluation import calculate_metrics
from interfaces.data.spiking_data_module import SpikeConverter
from plotting import plot_example_inference

def spike_rate_reg(y_hat):
    spike_rate = y_hat.sum(dim=0).mean()
    return 1.0 - spike_rate.item()


class MHBaseLitModel(pl.LightningModule):
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
        super().__init__()
        self.converter = None
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.alpha = alpha
        self.beta = beta
        self.head_width = head_width
        self.head_stride = head_stride
        self.num_hidden_layers = num_hidden_layers
        self.num_layers = self.num_hidden_layers + 2 # For Input and Output
        self.num_heads = (self.num_inputs - self.head_width) // self.head_stride + 1
        self.regularization = None
        self.learning_rate = learning_rate

        self.layers = self._init_layers()
        pass

    def _init_layers(self):
        heads = []
        for i in range(self.num_heads):
            curr_head = [nn.Linear(self.head_width, 2 * self.head_width, bias=False), snn.Synaptic(alpha=self.alpha, beta=self.beta, learn_threshold=True, learn_beta=True, learn_alpha=True, threshold=0.1)]
            heads.append(nn.Sequential(*curr_head))
        heads = nn.ModuleList(heads)
        layers = []
        for i in range(1, self.num_layers):
            if i == 1:
                layers.append(nn.Linear(2 * self.num_inputs, self.num_hidden, bias=False))
            elif i == self.num_layers - 1:
                layers.append(nn.Linear(self.num_hidden, self.num_outputs, bias=False))
            else:
                layers.append(nn.Linear(self.num_hidden, self.num_hidden, bias=False))
            layers.append(snn.Synaptic(alpha=self.alpha, beta=self.beta, threshold=0.1, learn_threshold=True, learn_beta=True, learn_alpha=True))
        return torch.nn.ModuleList([heads, *layers])

    def set_converter(self, converter: SpikeConverter):
        self.converter = converter

    def calc_accuracy(self, y_hat, y):
        score = balanced_accuracy_score(y_hat.flatten(), y.flatten())
        self.log("accuracy", score)

    def _init_membranes(self):
        membranes = []
        head_membranes = []
        for i in range(self.num_heads):
            head_membranes.append(self.layers[0][i][1].init_synaptic())
        membranes.append(
            head_membranes
        )
        membranes.extend(lif.init_synaptic() for lif in self.layers[2:self.num_layers * 2:2])
        return membranes

    @abc.abstractmethod
    def forward(self, x):
        pass

    def calc_loss(self, y_hat, y):
        loss = self.loss(y_hat, y)
        if self.regularization:
            for reg_method in self.regularization:
                loss += reg_method(y_hat)
        return loss

    def calculate_gradient_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def training_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _ = self(x)
        loss = self.calc_loss(spike_hat, y)
        self.log("train_loss", loss, sync_dist=True)
        grad_norm = self.calculate_gradient_norm()
        self.log("grad_norm", grad_norm, sync_dist=True)
        spike_rate = spike_hat.sum().item()
        self.log("spike_rate", spike_rate, sync_dist=True)
        # Add dual-modal regularization loss
        gmm_reg = torch.tensor(0.0)
        for head in self.layers[0]:
            gmm = GaussianMixture(n_components=2).fit(head[0].weight.detach().cpu().numpy().reshape(-1, 1))
            means = torch.tensor(gmm.means_.flatten(), dtype=torch.float32, device=head[0].weight.device)
            variances = torch.tensor(gmm.covariances_.flatten(), dtype=torch.float32, device=head[0].weight.device)

            # Regularization loss: maximize mean separation and minimize variance
            l_reg = -torch.abs(means[0] - means[1]) - (variances[0] + variances[1])

            # Normalize by the number of modules (optional)
            gmm_reg += l_reg / len(self.layers[0])
        self.log("gmm_reg", gmm_reg, sync_dist=True)
        return loss + gmm_reg

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
        self.log("val_loss", loss, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, _ = self(x)
        # Convert output to true output
        output_pred = self.converter.decode_inference(spike_hat.detach().cpu().numpy())
        accuracy, mse, auroc, auprc, f1 = calculate_metrics(
            y.detach().cpu().numpy(), output_pred
        )
        self.log("test_accuracy", accuracy, sync_dist=True)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_auroc", auroc, sync_dist=True)
        self.log("test_auprc", auprc, sync_dist=True)
        self.log("test_f1", f1, sync_dist=True)
        return accuracy, mse, auroc, auprc, f1

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


class MHLitModel(MHBaseLitModel):
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

    def _infer_slice(self, x, membranes):
        spike = None
        spike_counts = []
        # Deal with head layer first
        head_membranes = membranes[0]
        head_layer = self.layers[0]
        chunks = x.unfold(dimension=-1, size=self.head_width, step=self.head_stride)
        outputs = []
        for i, head in enumerate(head_layer):
            curr = head[0](chunks[:, :, i, :])
            spike, syn_mem, mem_mem = head[1](curr, head_membranes[i][0], head_membranes[i][1])
            head_membranes[i] = (syn_mem, mem_mem)
            outputs.append(spike)
        membranes[0] = head_membranes
        output = torch.cat(outputs, dim=-1)
        for n in range(1, self.num_layers):
            curr = self.layers[((n-1) * 2) + 1](output)
            spike, syn_mem, mem_mem = self.layers[n * 2](curr, membranes[n][0], membranes[n][1])
            membranes[n] = (syn_mem, mem_mem)
            output = spike
            spike_counts.append(torch.count_nonzero(spike).item())
        return spike, membranes[-1], spike_counts

    def forward(self, x):
        full_spike = []
        spike_recordings = []
        # x -> [N x exp x C x freq x time]
        membranes = self._init_membranes()
        num_polarizations = x.shape[2]
        if num_polarizations != 1:
            x = torch.flatten(x, 2, 3).unsqueeze(2)
        for t in range(x.shape[-1]):
            data = x[:, :, 0, :, t]
            spike, _, spike_rec = self._infer_slice(data, membranes)
            spike_recordings.append(spike_rec)
            if num_polarizations != 1 and self.num_outputs == self.num_inputs:  # Only catch polarized output
                spike = spike.view(*(spike.shape[:-1]), num_polarizations, -1)
            full_spike.append(spike)
        full_spike = torch.stack(full_spike, dim=0)  # [time x N x exp x C x freq]
        full_spike = torch.moveaxis(full_spike, 0, -1)
        if num_polarizations == 1 or self.num_outputs != self.num_inputs:
            full_spike = full_spike.unsqueeze(2)
        return torch.moveaxis(full_spike, 0, 1), spike_recordings
