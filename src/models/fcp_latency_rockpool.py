from torch.nn import MSELoss
from interfaces.models.model_rockpool import LitModelPatchedRockpool


class LitFcLatencyPatchedRockpool(LitModelPatchedRockpool):
    def __init__(self,
                 num_inputs: int,
                 num_hidden: int,
                 num_outputs: int,
                 num_layers: int):
        super().__init__(num_inputs, num_hidden, num_outputs, num_layers)
        self.loss = MSELoss()
        self.float()
        self.save_hyperparameters()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        spike_hat, mem_hat = self(x)
        loss = self.calc_loss(spike_hat, y)
        self.log("val_loss", loss, sync_dist=True)
