"""
A non-converting encoder and decoder for ANN inference, which does not work with spikes.
"""

import numpy as np

from interfaces.data.spiking_data_module import SpikeConverter


class NonConverter(SpikeConverter):
    """
    Exists for ANN inference without changing downstream implementation
    """

    def __init__(self, patched=False):
        self.patched = patched
        super().__init__()

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        if self.patched:
            final_size = x_data.shape[-1] * x_data.shape[-2]
            return x_data.astype("float32").reshape((*(x_data.shape[:-2]), final_size))
        return x_data.astype("float32")

    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        if self.patched:
            final_size = y_data.shape[-1] * y_data.shape[-2]
            return y_data.astype("float32").reshape((*(y_data.shape[:-2]), final_size))
        return y_data.astype("float32")

    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        return inference
