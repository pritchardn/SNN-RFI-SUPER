"""
Rate-coding spike encoder and decoder.
"""

import numpy as np
import torch
from snntorch import spikegen

from interfaces.data.spiking_data_module import SpikeConverter


class RateSpikeConverter(SpikeConverter):
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
        out = y_data.squeeze(1)  # N x freq x time
        out = np.moveaxis(out, 1, 2)  # N x time x freq
        pad = y_data.shape[-1]
        real_output = np.full_like(out, -1)
        for i in range(out.shape[0]):
            example = out[i]
            curr_out = []
            for t in range(example.shape[0]):
                target_freqs = np.argwhere(example[t] == 1)
                curr_out.append(
                    np.append(target_freqs, [-1] * (pad - len(target_freqs)))
                )
            real_output[i] = np.array(curr_out)
        return real_output  # N x time x freq !IMPORTANT

    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        return inference.mean(axis=0) > 0.75  # Assuming [exp, N, C, freq, time]
        # Could also explore taking average > 0.5, or any spiking pixels at all.
