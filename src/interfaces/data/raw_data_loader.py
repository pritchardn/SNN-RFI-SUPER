from abc import ABC, abstractmethod

import numpy as np


class RawDataLoader(ABC):
    def __init__(self, data_dir: str):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.rfi_models = []
        self.data_dir = data_dir

    @abstractmethod
    def load_data(self, excluded_rfi=None):
        pass

    def fetch_train_x(self) -> np.ndarray:
        return self.train_x

    def fetch_train_y(self) -> np.ndarray:
        return self.train_y

    def fetch_test_x(self) -> np.ndarray:
        return self.test_x

    def fetch_test_y(self) -> np.ndarray:
        return self.test_y
