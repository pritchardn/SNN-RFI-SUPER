import os
import pickle
from typing import Union

import numpy as np

from interfaces.data.raw_data_loader import RawDataLoader


def _normalize(
    image_data: np.ndarray, masks: np.ndarray, min_threshold: int, max_threshold: int
) -> np.ndarray:
    _max = np.mean(image_data[np.invert(masks)]) + max_threshold * np.std(
        image_data[np.invert(masks)]
    )
    _min = np.absolute(
        np.mean(image_data[np.invert(masks)])
        - min_threshold * np.std(image_data[np.invert(masks)])
    )
    image_data = np.clip(image_data, _min, _max)
    image_data = np.log(image_data)
    # Rescale
    minimum, maximum = np.min(image_data), np.max(image_data)
    image_data = (image_data - minimum) / (maximum - minimum)
    return image_data


class HeraDataLoader(RawDataLoader):
    def _prepare_data(self):
        self.train_x[self.train_x == np.inf] = np.finfo(self.train_x.dtype).max
        self.test_x[self.test_x == np.inf] = np.finfo(self.test_x.dtype).max
        self.test_x = self.test_x.astype("float32")
        self.train_x = self.train_x.astype("float32")
        self.test_x = _normalize(self.test_x, self.test_y, 1, 4)
        self.train_x = _normalize(self.train_x, self.train_y, 1, 4)
        self.convert_pytorch()
        self.val_x = self.test_x.copy()
        self.val_y = self.test_y.copy()
        self.limit_datasets()

    def load_data(self, excluded_rfi: Union[str, None] = None):
        if excluded_rfi is None:
            rfi_models = []
            file_path = os.path.join(self.data_dir, "HERA_04-03-2022_all.pkl")
            train_x, train_y, test_x, test_y = np.load(file_path, allow_pickle=True)
        else:
            rfi_models = ["rfi_stations", "rfi_dtv", "rfi_impulse", "rfi_scatter"]
            rfi_models.remove(excluded_rfi)
            test_file_path = os.path.join(
                self.data_dir, f"HERA_04-03-2022_{excluded_rfi}.pkl"
            )
            _, _, test_x, test_y = np.load(test_file_path, allow_pickle=True)

            train_file_path = os.path.join(
                self.data_dir, f'HERA_04-03-2022_{"-".join(rfi_models)}.pkl'
            )
            train_x, train_y, _, _ = np.load(train_file_path, allow_pickle=True)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.rfi_models = rfi_models
        self.original_size = self.train_x.shape[1]
        self._prepare_data()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()


class LofarDataLoader(RawDataLoader):
    def _prepare_data(self):
        self.train_x[self.train_x == np.inf] = np.finfo(self.train_x.dtype).max
        self.test_x[self.test_x == np.inf] = np.finfo(self.test_x.dtype).max
        self.test_x = self.test_x.astype("float32")
        self.train_x = self.train_x.astype("float32")
        self.test_x = _normalize(self.test_x, self.test_y, 3, 95)
        self.train_x = _normalize(self.train_x, self.train_y, 3, 95)
        self.convert_pytorch()
        self.val_x = self.test_x.copy()
        self.val_y = self.test_y.copy()
        self.limit_datasets()

    def load_data(self, excluded_rfi=None):
        filepath = os.path.join(self.data_dir, "LOFAR_Full_RFI_dataset.pkl")
        print(f"Loading LOFAR data from {filepath}")
        with open(filepath, "rb") as ifile:
            train_x, train_y, test_x, test_y = pickle.load(ifile)
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y
        self.original_size = self.train_x.shape[1]
        self._prepare_data()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        # self.filter_noiseless_val_patches()
        # self.filter_noiseless_train_patches()
