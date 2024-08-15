"""
Delta-modulation spike encoder and decoder.
"""

import numpy as np
import torch
from snntorch import spikegen

from data.data_loaders import HeraDataLoader
from data.utils import decode_delta_inference
from interfaces.data.spiking_data_module import SpikeConverter


class DeltaSpikeConverter(SpikeConverter):
    def __init__(self, threshold: float, off_spikes: bool):
        self.threshold = threshold
        self.off_spikes = off_spikes

    def _encode_x_off_spikes(self, x_data: np.ndarray) -> np.ndarray:
        encoded = spikegen.delta(
            torch.from_numpy(np.moveaxis(x_data, -1, 0)),
            threshold=self.threshold,
            off_spike=True,
        ).numpy()
        encoded = np.moveaxis(encoded, 0, -1)
        return np.expand_dims(encoded, axis=1)  # [N, exp, C, freq, time]

    def _encode_x_on_spikes(self, x_data: np.ndarray) -> np.ndarray:
        return self.encode_y(x_data)

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        if self.off_spikes:
            return self._encode_x_off_spikes(x_data)
        return self._encode_x_on_spikes(x_data)

    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        """
        Returns a [1 x N x C x freq * 2 x time] array. All outputs are positive spikes.
        Algorithm copies each frequency channel adjacent to each other.
        -1s in the first copy are erased
        1s in the second copy are erased
        -1s in the second copy are replaced with 1s
        :param y_data:
        :return: An array of purely positive spikes where even-indexed frequency channels contain
        on spike, and odd-indexed frequency channels contain off spikes.
        """
        original_encoded = self._encode_x_off_spikes(y_data)
        out = np.zeros(
            (y_data.shape[0], 1, y_data.shape[1], y_data.shape[2] * 2, y_data.shape[3]),
            dtype=y_data.dtype,
        )
        # Make two copies of encoded data in frequency dimension
        out[:, :, :, ::2, :] = original_encoded.copy()
        out[:, :, :, 1::2, :] = original_encoded.copy()
        # Erase off spikes in first copy
        out[:, :, :, ::2, :] = np.where(
            out[:, :, :, ::2, :] == -1, 0, out[:, :, :, ::2, :]
        )
        # Erase on spikes in second copy
        out[:, :, :, 1::2, :] = np.where(
            out[:, :, :, 1::2, :] == 1, 0, out[:, :, :, 1::2, :]
        )
        # Replace off spikes in second copy with on spikes
        out[:, :, :, 1::2, :] = np.where(
            out[:, :, :, 1::2, :] == -1, 1, out[:, :, :, 1::2, :]
        )
        return out

    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        return decode_delta_inference(inference, use_numpy=True)


if __name__ == "__main__":
    # Load HERA data
    data_source = HeraDataLoader("./data", patch_size=32, stride=32, limit=0.1)
    print("Loaded data")
    # Setup encoder
    encoder = DeltaSpikeConverter(threshold=0.1, off_spikes=True)
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
