"""
Latency spike encoder and decoder.
"""

import numpy as np
import torch
from snntorch import spikegen

from interfaces.data.spiking_data_module import SpikeConverter


class LatencySpikeConverter(SpikeConverter):
    def __init__(self, exposure: int, tau: float, normalize: bool):
        self.exposure = exposure
        self.tau = tau
        self.normalize = normalize

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        out_shape = (
            x_data.shape[0],
            self.exposure,
            x_data.shape[1],
            x_data.shape[2],
            x_data.shape[3],
        )
        output = np.zeros(out_shape, dtype=x_data.dtype)
        for i, frame in enumerate(x_data):
            frame = torch.from_numpy(np.moveaxis(frame, 0, -1))
            frame = spikegen.latency(
                frame, num_steps=self.exposure, tau=self.tau, normalize=True
            )
            frame = np.moveaxis(frame.numpy(), -1, 1)
            frame[self.exposure - 1, ...] = 0
            output[i] = frame
        return output.astype("float32")

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
