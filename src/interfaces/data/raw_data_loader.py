"""
Interface for loading raw data.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

from data.utils import extract_patches, filter_noiseless_patches


def calc_limit_int(limit: float, data_len: int) -> int:
    if limit is None:
        return data_len
    return int(data_len * limit)


class RawDataLoader(ABC):
    def __init__(
            self,
            data_dir: str,
            limit: float = None,
            patch_size: int = None,
            stride: int = None,
    ):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None
        self.val_x = None
        self.val_y = None
        self.rfi_models = []
        self.original_size = None
        self.data_dir = data_dir
        self.limit = limit
        self.patch_size = patch_size
        self.stride = stride if stride else patch_size

    @abstractmethod
    def load_data(self, *args):
        pass

    def fetch_train_x(self) -> np.ndarray:
        return self.train_x

    def fetch_train_y(self) -> np.ndarray:
        return self.train_y

    def fetch_test_x(self) -> np.ndarray:
        return self.test_x

    def fetch_test_y(self) -> np.ndarray:
        return self.test_y

    def fetch_val_x(self) -> np.ndarray:
        return self.val_x

    def fetch_val_y(self) -> np.ndarray:
        return self.val_y

    def create_patches(self, patch_size: int, stride: int):
        self.train_x = extract_patches(
            torch.from_numpy(self.train_x), patch_size, stride
        ).numpy()
        self.train_y = extract_patches(
            torch.from_numpy(self.train_y), patch_size, stride
        ).numpy()
        self.test_x = extract_patches(
            torch.from_numpy(self.test_x), patch_size, stride
        ).numpy()
        self.test_y = extract_patches(
            torch.from_numpy(self.test_y), patch_size, stride
        ).numpy()
        self.val_x = extract_patches(
            torch.from_numpy(self.val_x), patch_size, stride
        ).numpy()
        self.val_y = extract_patches(
            torch.from_numpy(self.val_y), patch_size, stride
        ).numpy()

    def limit_datasets(self):
        limit = calc_limit_int(self.limit, len(self.train_x))
        self.train_x = self.train_x[:limit]
        self.train_y = self.train_y[:limit]

        limit = calc_limit_int(self.limit, len(self.test_x))
        self.test_x = self.test_x[:limit]
        self.test_y = self.test_y[:limit]
        limit = calc_limit_int(self.limit, len(self.val_x))
        self.val_x = self.val_x[:limit]
        self.val_y = self.val_y[:limit]

    def filter_noiseless_val_patches(self):
        self.val_x, self.val_y = filter_noiseless_patches(self.val_x, self.val_y)

    def filter_noiseless_train_patches(self):
        self.train_x, self.train_y = filter_noiseless_patches(
            self.train_x, self.train_y
        )

    def convert_pytorch(self):
        self.train_x = np.moveaxis(self.train_x, -1, 1)
        self.train_y = np.moveaxis(self.train_y, -1, 1).astype("float32")
        self.test_x = np.moveaxis(self.test_x, -1, 1)
        self.test_y = np.moveaxis(self.test_y, -1, 1).astype("float32")

    def release_memory(self):
        del self.train_x
        del self.train_y
        del self.val_x
        del self.val_y
