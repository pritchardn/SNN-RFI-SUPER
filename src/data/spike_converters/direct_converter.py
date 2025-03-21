"""
Implements a direct converter that interprets the original data as current to be
injected at each time step.

Decoding is to be implemented as in rate conversion.
"""

import numpy as np
import torch
from snntorch import spikegen

from interfaces.data.spiking_data_module import SpikeConverter


class DirectSpikeConverter(SpikeConverter):
    def __init__(self, exposure: int):
        self.exposure = exposure

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        encoded = np.expand_dims(x_data, axis=1)
        encoded = np.repeat(encoded, self.exposure, axis=1)
        return encoded

    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        output_timings = np.zeros(y_data.shape, dtype=y_data.dtype)
        output_timings[y_data > 0] = 0
        output_timings[y_data == 0] = self.exposure - 1
        return output_timings

    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        """
        Rebuilds mask from spiking output. In this case, any pixel location that did not spike until
        the end of the exposure time is considered the background.
        Assumes a shape of [N, exposure, C, freq, time]
        :return: [N, C, freq, time]
        """
        return inference[:-1, :, :, :, :].sum(axis=0)
