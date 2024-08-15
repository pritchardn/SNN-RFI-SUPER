"""
Delta-modulation spike encoder and decoder with exposure.
"""

import numpy as np
import torch

from snntorch import spikegen
from data.data_loaders import HeraDataLoader
from interfaces.data.spiking_data_module import SpikeConverter


class DeltaExposureSpikeConverter(SpikeConverter):
    def __init__(self, threshold: float, exposure: int):
        self.threshold = threshold
        self.exposure = exposure

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        # Generate single exposure encoding
        encoded = spikegen.delta(
            torch.from_numpy(np.moveaxis(x_data, -1, 0)),
            threshold=self.threshold,
            off_spike=True,
            padding=True,
        ).numpy()
        encoded = np.moveaxis(encoded, 0, -1)
        encoded = np.expand_dims(encoded, axis=1)  # [N, exp, C, freq, time]
        encoded = np.pad(
            encoded,
            ((0, 0), (0, self.exposure - 1), (0, 0), (0, 0), (0, 0)),
            "constant",
        )
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
        Rebuilds the mask in the same way as latency encoding
        """
        return inference[:-1, :, :, :, :].sum(axis=0)


if __name__ == "__main__":
    # Load HERA data
    data_source = HeraDataLoader("./data", patch_size=32, stride=32, limit=0.1)
    print("Loaded data")
    # Setup encoder
    encoder = DeltaExposureSpikeConverter(threshold=0.1, exposure=6)
    # Convert X
    data_source.load_data()
    train_x = torch.from_numpy(encoder.encode_x(data_source.fetch_train_x()))
    # Convert y
    train_y = torch.from_numpy(encoder.encode_y(data_source.fetch_train_y()))
    print("Conversions done")
    # Decode Y-conversion
    inference = np.moveaxis(train_y.numpy(), 1, 0)
    decoded_y = encoder.decode_inference(inference)
    print("Decoding done")
    # Check decoded Y matches original
    assert np.allclose(decoded_y, data_source.fetch_train_y())
