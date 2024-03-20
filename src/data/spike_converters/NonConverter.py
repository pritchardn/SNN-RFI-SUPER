import numpy as np

from interfaces.data.spiking_data_module import SpikeConverter


class NonConverter(SpikeConverter):
    """
    Exists for ANN inference without changing downstream implementation
    """

    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        return x_data.astype("float32")

    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        return y_data.astype("float32")

    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        return inference
