import os
import pickle
from typing import Union

import numpy as np

from interfaces.data.raw_data_loader import RawDataLoader


class HeraDataLoader(RawDataLoader):

    def _normalize(self, image_data: np.ndarray, masks: np.ndarray) -> np.ndarray:
        max_threshold = 4
        min_threshold = 1
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

    def _convert_pytorch(self):
        self.train_x = np.moveaxis(self.train_x, -1, 1).astype(np.float32)
        self.train_y = np.moveaxis(self.train_y, -1, 1).astype(np.float32)
        self.test_x = np.moveaxis(self.test_x, -1, 1).astype(np.float32)
        self.test_y = np.moveaxis(self.test_y, -1, 1).astype(np.float32)

    def _prepare_data(self):
        self.train_x[self.train_x == np.inf] = np.finfo(self.train_x.dtype).max
        self.test_x[self.test_x == np.inf] = np.finfo(self.test_x.dtype).max
        self.test_x = self.test_x.astype("float32")
        self.train_x = self.train_x.astype("float32")
        self.test_x = self._normalize(self.test_x, self.test_y)
        self.train_x = self._normalize(self.train_x, self.train_y)
        self._convert_pytorch()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)

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
        self._prepare_data()


class LofarDataLoader(RawDataLoader):

    def load_data(self, excluded_rfi=None):
        filepath = os.path.join(self.data_dir, "LOFAR_Full_RFI_dataset.pkl")
        print(f"Loading LOFAR data from {filepath}")
        with open(filepath, "rb") as ifile:
            train_x, train_y, test_x, test_y = pickle.load(ifile)
            self.train_x = train_x
            self.train_y = train_y
            self.test_x = test_x
            self.test_y = test_y


if __name__ == "__main__":
    print("Testing data loading")
    hera_data_loader = HeraDataLoader("./data")
    hera_data_loader.load_data()
    assert hera_data_loader.train_x is not None
    assert hera_data_loader.train_y is not None
    assert hera_data_loader.test_x is not None
    assert hera_data_loader.test_y is not None
    del hera_data_loader
    lofar_data_loader = LofarDataLoader("./data")
    lofar_data_loader.load_data()
    assert lofar_data_loader.train_x is not None
    assert lofar_data_loader.train_y is not None
    assert lofar_data_loader.test_x is not None
    assert lofar_data_loader.test_y is not None
    del lofar_data_loader
