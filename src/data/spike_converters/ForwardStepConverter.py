import numpy as np
import torch

from data.data_loaders import HeraDataLoader
from interfaces.data.spiking_data_module import SpikeConverter


class ForwardStepConverter(SpikeConverter):

    def __init__(self, threshold: float, exposure: int):
        self.threshold = threshold
        self.exposure = exposure

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        """
        Inspired by the step-forward algorithm seen in Speech2Spikes Stewart et al. (2023)
        https://dl.acm.org/doi/10.1145/3584954.3584995
        """
        out = np.zeros(
            (x_data.shape[0], 1, x_data.shape[1], x_data.shape[2] * 2, x_data.shape[3]))
        for i, frame in enumerate(x_data):
            frame = np.squeeze(frame, axis=0)
            levels = np.round(frame, 0)
            cumsum = np.cumsum(frame, axis=-1)
            interim = np.vstack((frame, cumsum))
            for t in range(frame.shape[-1]):
                curr = (interim[..., t] - levels > self.threshold).astype(int) - (
                        interim[..., t] - levels < -self.threshold).astype(int)
                out[i, :, :, :, t] = curr
                levels += out[i, ..., t] * self.threshold
        return out

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
    encoder = ForwardStepConverter(threshold=0.1, exposure=2)
    # Convert X
    data_source.load_data()
    train_x = torch.from_numpy(encoder.encode_x(data_source.fetch_train_x()))
    # Convert y
    train_y = torch.from_numpy(encoder.encode_y(data_source.fetch_train_y()))
    print("Conversions done")
