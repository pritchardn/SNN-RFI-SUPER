"""
Contains implemented data loaders for various radio astronomy datasets.
"""

import os
import pickle
from typing import Union

import h5py
import numpy as np
import sklearn.model_selection
from tqdm import tqdm

from interfaces.data.raw_data_loader import RawDataLoader


def _delta_normalize(image_data: np.ndarray, kernel_size=3) -> np.ndarray:
    print("Performing Delta-Normalization")
    print(f"Original mean: {np.mean(image_data)}")
    for i, frame in tqdm(enumerate(image_data)):
        output_frame = np.zeros_like(frame)
        for tstep in range(1, frame.shape[0]):  # Ignore first timestep
            for frequency in range(frame.shape[1]):
                neighbouring_activity = 0.0
                min_freq = max(0, frequency - kernel_size // 2)
                max_freq = min(frame.shape[1] - 1, frequency + kernel_size // 2)
                count_elem = 0
                for j in range(min_freq, max_freq + 1):
                    neighbouring_activity += frame[tstep - 1, j]
                    count_elem += 1
                output_frame[tstep, frequency] = max(0, frame[
                    tstep, frequency] - neighbouring_activity / count_elem)
        image_data[i] = output_frame
    print(f"New mean : {np.mean(image_data)}")
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
        self.test_x = _delta_normalize(self.test_x)
        self.train_x = _delta_normalize(self.train_x)
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


class TabascalDataLoader(RawDataLoader):
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

    def load_data(self, num_sat: int = 2, num_ground: int = 3):
        filepath = os.path.join(
            self.data_dir,
            f"obs_100AST_{num_sat}SAT_{num_ground}"
            f"GRD_512BSL_64A_512T-0440-1462_016I_512F-1.227e+09-1.334e+09.hdf5",
        )
        print(f"Loading Tabascal data from {filepath}")
        h5file = h5py.File(filepath, "r")
        image_data = h5file.get("vis")
        image_data = np.array(image_data)
        mask_fieldname = "masks_median"
        masks = h5file.get(mask_fieldname)
        masks = np.array(masks).astype("bool")
        train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
            image_data, masks, test_size=0.2
        )
        h5file.close()
        self.train_x = np.moveaxisd(train_x, 1, 2)
        self.train_y = np.moveaxis(train_y, 1, 2)
        self.test_x = np.moveaxis(test_x, 1, 2)
        self.test_y = np.moveaxis(test_y, 1, 2)
        self.original_size = self.train_x.shape[1]
        self._prepare_data()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        # self.filter_noiseless_val_patches()
        # self.filter_noiseless_train_patches()
