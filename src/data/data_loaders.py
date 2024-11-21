"""
Contains implemented data loaders for various radio astronomy datasets.
"""

import os
import pickle
from typing import Union

import numpy as np
from tqdm import tqdm

from data.utils import test_train_split, extract_polarization
from interfaces.data.raw_data_loader import RawDataLoader


def _delta_normalize(image_data: np.ndarray, kernel_size=3) -> np.ndarray:
    image_data = np.moveaxis(image_data, 1, 2)
    print("Performing Delta-Normalization")
    print(f"Original mean: {np.mean(image_data)}")
    for i, frame in tqdm(enumerate(image_data)):
        output_frame = np.zeros_like(frame)
        for tstep in range(1, frame.shape[0]):  # Ignore first timestep
            step_back = min(tstep, 1)
            for frequency in range(frame.shape[1]):
                neighbouring_activity = 0.0
                min_freq = max(0, frequency - kernel_size // 2)
                max_freq = min(frame.shape[1] - 1, frequency + kernel_size // 2)
                count_elem = 0
                for j in range(min_freq, max_freq + 1):
                    neighbouring_activity += frame[tstep - step_back, j]
                    count_elem += 1
                output_frame[tstep, frequency] = max(
                    0.0, frame[tstep, frequency] - neighbouring_activity / count_elem
                )
        image_data[i] = output_frame
    print(f"New mean : {np.mean(image_data)}")
    image_data = np.moveaxis(image_data, 1, 2)
    return image_data


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
            file_path = os.path.join(self.data_dir, "HERA_21-11-2024_all.pkl")
            data, _, masks = np.load(file_path, allow_pickle=True)
            train_x, train_y, test_x, test_y = test_train_split(data, masks)
            train_x, train_y = extract_polarization(train_x, train_y, 0)
            test_x, test_y = extract_polarization(test_x, test_y, 0)
        else:
            raise NotImplementedError("Excluded RFI not implemented for HERA dataset with Polarization")
        self.train_x = np.moveaxis(train_x, 1, 2)
        self.train_y = np.moveaxis(train_y, 1, 2)
        self.test_x = np.moveaxis(test_x, 1, 2)
        self.test_y = np.moveaxis(test_y, 1, 2)
        self.rfi_models = rfi_models
        self.original_size = self.train_x.shape[1]
        self._prepare_data()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()


class HeraDeltaNormLoader(RawDataLoader):
    def load_data(self):
        file_path = os.path.join(self.data_dir, "HERA-04-03-2022_all_delta_norm.pkl")
        train_x, train_y, test_x, test_y, val_x, val_y = np.load(
            file_path, allow_pickle=True
        )
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.val_x = val_x
        self.val_y = val_y
        self.limit_datasets()
        self.original_size = self.train_x.shape[-1]
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()


class HeraPolarizationDataLoader(RawDataLoader):
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
            file_path = os.path.join(self.data_dir, "HERA_21-11-2024_all.pkl")
            data, _, masks = np.load(file_path, allow_pickle=True)
            train_x, train_y, test_x, test_y = test_train_split(data, masks)
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
        self.train_x = np.moveaxis(train_x, 1, 2)
        self.train_y = np.moveaxis(train_y, 1, 2)
        self.test_x = np.moveaxis(test_x, 1, 2)
        self.test_y = np.moveaxis(test_y, 1, 2)
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

    def load_data(self):
        filepath = os.path.join(self.data_dir, "LOFAR_Full_RFI_dataset.pkl")
        print(f"Loading LOFAR data from {filepath}")
        with open(filepath, "rb") as ifile:
            train_x, train_y, test_x, test_y = pickle.load(ifile)
        self.train_x = np.moveaxis(train_x, 1, 2)
        self.train_y = np.moveaxis(train_y, 1, 2)
        self.test_x = np.moveaxis(test_x, 1, 2)
        self.test_y = np.moveaxis(test_y, 1, 2)
        self.original_size = self.train_x.shape[1]
        self._prepare_data()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()


class LofarDeltaNormLoader(RawDataLoader):
    def load_data(self):
        file_path = os.path.join(self.data_dir, "LOFAR_Full_RFI_dataset_delta_norm.pkl")
        train_x, train_y, test_x, test_y, val_x, val_y = np.load(
            file_path, allow_pickle=True
        )
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.val_x = val_x
        self.val_y = val_y
        self.limit_datasets()
        self.original_size = self.train_x.shape[-1]
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_train_patches()


def create_delta_normalized_hera():
    file_path = os.path.join("data", "HERA_04-03-2022_all.pkl")
    train_x, train_y, test_x, test_y = np.load(file_path, allow_pickle=True)
    train_x = np.moveaxis(train_x, 1, 2)
    train_y = np.moveaxis(train_y, 1, 2)
    test_x = np.moveaxis(test_x, 1, 2)
    test_y = np.moveaxis(test_y, 1, 2)
    train_x[train_x == np.inf] = np.finfo(train_x.dtype).max
    test_x[test_x == np.inf] = np.finfo(test_x.dtype).max
    test_x = test_x.astype("float32")
    train_x = train_x.astype("float32")
    test_x = _normalize(test_x, test_y, 1, 4)
    train_x = _normalize(train_x, train_y, 1, 4)
    train_x = _delta_normalize(train_x)
    test_x = _delta_normalize(test_x)
    train_x = np.moveaxis(train_x, -1, 1).astype(np.float32)
    train_y = np.moveaxis(train_y, -1, 1).astype(np.float32)
    test_x = np.moveaxis(test_x, -1, 1).astype(np.float32)
    test_y = np.moveaxis(test_y, -1, 1).astype(np.float32)
    val_x = test_x.copy()
    val_y = test_y.copy()
    file_path = os.path.join("data", "HERA-04-03-2022_all_delta_norm.pkl")
    with open(file_path, "wb") as ofile:
        pickle.dump([train_x, train_y, test_x, test_y, val_x, val_y], ofile)


def create_delta_normalized_lofar():
    filepath = os.path.join("./data", "LOFAR_Full_RFI_dataset.pkl")
    print(f"Loading LOFAR data from {filepath}")
    with open(filepath, "rb") as ifile:
        train_x, train_y, test_x, test_y = pickle.load(ifile)
        train_x = np.moveaxis(train_x, 1, 2)
        train_y = np.moveaxis(train_y, 1, 2)
        test_x = np.moveaxis(test_x, 1, 2)
        test_y = np.moveaxis(test_y, 1, 2)
    train_x[train_x == np.inf] = np.finfo(train_x.dtype).max
    test_x[test_x == np.inf] = np.finfo(test_x.dtype).max
    test_x = test_x.astype("float32")
    train_x = train_x.astype("float32")
    test_x = _normalize(test_x, test_y, 3, 95)
    train_x = _normalize(train_x, train_y, 3, 95)
    test_x = _delta_normalize(test_x)
    train_x = _delta_normalize(train_x)
    train_x = np.moveaxis(train_x, -1, 1).astype(np.float32)
    train_y = np.moveaxis(train_y, -1, 1).astype(np.float32)
    test_x = np.moveaxis(test_x, -1, 1).astype(np.float32)
    test_y = np.moveaxis(test_y, -1, 1).astype(np.float32)
    val_x = test_x.copy()
    val_y = test_y.copy()
    file_path = os.path.join("data", "LOFAR_Full_RFI_dataset_delta_norm.pkl")
    with open(file_path, "wb") as ofile:
        pickle.dump([train_x, train_y, test_x, test_y, val_x, val_y], ofile)


def main():
    create_delta_normalized_hera()
    create_delta_normalized_lofar()


if __name__ == "__main__":
    main()
