"""
Rate-coding spike encoder and decoder.
"""

import numpy as np
import torch
from snntorch import spikegen

from interfaces.data.spiking_data_module import SpikeConverter


class RateFullSpikeConverter(SpikeConverter):
    def __init__(self, exposure: int):
        self.exposure = exposure

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        encoded = spikegen.rate(
            torch.from_numpy(x_data), num_steps=self.exposure
        ).numpy()
        return np.moveaxis(encoded, 0, 1)

    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        """
        Returns an [N x time x freq] array, where each entry in [N, T] is a -1 padded list of
        frequencies we treat as classification targets.
        This is a little confusing at first, but is done to make loss calculation easier.
        :param y_data:
        :return:
        """
        out = np.expand_dims(y_data, axis=1)
        out = np.repeat(out, self.exposure, axis=1)
        return out

    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        return inference.mean(axis=1) > 0.5  # Assuming [N, exp, C, freq, time]
        # Could also explore taking average > 0.5, or any spiking pixels at all.
