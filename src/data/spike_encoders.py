import torch
from snntorch import spikegen

from interfaces.data.spiking_data_module import SpikeConverter
import numpy as np


class LatencySpikeEncoder(SpikeConverter):

    def __init__(self, exposure: int, tau: float, normalize: bool):
        self.exposure = exposure
        self.tau = tau
        self.normalize = normalize

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        for i, frame in enumerate(x_data):
            frame = torch.from_numpy(np.moveaxis(frame, 0, -1))
            # TODO: Build proper output object
            frame = spikegen.latency(frame, num_steps=self.exposure, tau=self.tau, normalize=True)
            frame = np.moveaxis(frame.numpy(), -1, 1)
            x_data[i] = frame
        return x_data

    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        return y_data

    def plot_sample(
            self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass
