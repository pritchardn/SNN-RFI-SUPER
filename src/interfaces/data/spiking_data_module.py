"""
This module contains the interface for the spike data converter.
"""

from abc import ABC, abstractmethod

import numpy as np


class SpikeConverter(ABC):
    @abstractmethod
    def encode_x(self, x_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def encode_y(self, y_data: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def plot_sample(
        self, x_data: np.ndarray, y_data: np.ndarray, output_dir: str, num: int
    ):
        pass

    @abstractmethod
    def decode_inference(self, inference: np.ndarray) -> np.ndarray:
        pass
