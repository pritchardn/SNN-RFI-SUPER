from abc import ABC, abstractmethod

import numpy as np


def _calc_limit_int(limit: float, data_len: int) -> int:
    if limit is None:
        return data_len
    else:
        return int(data_len * limit)


class RawDataLoader(ABC):

    def __init__(self, data_dir: str, limit: float = None):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.rfi_models = []
        self.data_dir = data_dir
        self.limit = limit

    @abstractmethod
    def load_data(self, excluded_rfi=None):
        pass

    def fetch_train_x(self) -> np.ndarray:
        limit = _calc_limit_int(self.limit, self.train_x.shape[0])
        return self.train_x[:limit]

    def fetch_train_y(self) -> np.ndarray:
        limit = _calc_limit_int(self.limit, self.train_y.shape[0])
        return self.train_y[:limit]

    def fetch_test_x(self) -> np.ndarray:
        limit = _calc_limit_int(self.limit, self.test_x.shape[0])
        return self.test_x[:limit]

    def fetch_test_y(self) -> np.ndarray:
        limit = _calc_limit_int(self.limit, self.test_y.shape[0])
        return self.test_y[:limit]
