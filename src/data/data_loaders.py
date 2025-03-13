"""
Contains implemented data loaders for various radio astronomy datasets.
"""
import os
import pickle
from typing import Union

import numpy as np
from tqdm import tqdm

from data.utils import test_train_split, extract_polarization
from interfaces.data.raw_data_loader import RawDataLoader, calc_limit_int


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
                if neighbouring_activity.shape[0] != 0:
                    if neighbouring_activity.dtype == np.complex64:
                        for z, val in enumerate(neighbouring_activity):
                            temp = frame[tstep, frequency, z] - val / count_elem
                            ex_real = max(0.0, temp.real)
                            ex_imag = max(0.0, temp.imag)
                            output_frame[tstep, frequency, z] = complex(ex_real, ex_imag)
                    else:
                        for z, val in enumerate(neighbouring_activity):
                            output_frame[tstep, frequency, z] = max(0.0, frame[
                                tstep, frequency, z] - val / count_elem)
                else:
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
    normalized_data = (image_data - np.mean(image_data, axis=0)) / np.std(image_data, axis=0)
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
            file_path = os.path.join(self.data_dir, "HERA-21-11-2024_all.pkl")
            data, _, masks = np.load(file_path, allow_pickle=True)
            train_x, train_y, test_x, test_y = test_train_split(data, masks)
        else:
            rfi_models = ["rfi_stations", "rfi_dtv", "rfi_impulse", "rfi_scatter"]
            rfi_models.remove(excluded_rfi)
            test_file_path = os.path.join(
                self.data_dir, f"HERA_21-11-2024_{excluded_rfi}.pkl"
            )
            _, _, test_x, test_y = np.load(test_file_path, allow_pickle=True)

            train_file_path = os.path.join(
                self.data_dir, f'HERA_21-11-2024_{"-".join(rfi_models)}.pkl'
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


class HeraDeltaNormLoader(RawDataLoader):
    def load_data(self):
        file_path = os.path.join(self.data_dir, "HERA-21-11-2024_all_delta_norm.pkl")
        data, masks = np.load(
            file_path, allow_pickle=True
        )
        data = np.expand_dims(data[:, 0, :, :], 1)
        masks = np.expand_dims(masks[:, 0, :, :], 1)
        train_x, train_y, test_x, test_y = test_train_split(data, masks)
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.train_x, self.train_y, self.val_x, self.val_y = test_train_split(self.train_x, self.train_y)
        self.limit_datasets()
        self.original_size = self.train_x.shape[-1]
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()
        pass


def calculate_dop(xx, yy):
    g0 = np.abs(xx) ** 2 + np.abs(yy) ** 2
    g1 = np.abs(xx) ** 2 - np.abs(yy) ** 2
    g2 = 2 * np.real(yy * np.conj(xx))
    g3 = 2 * np.imag(yy * np.conj(xx))
    out = np.zeros((xx.shape[0], 1, xx.shape[-1]))
    for n in range(xx.shape[0]):
        for t in range(xx.shape[-1]):
            denominator = np.average(g0[n, :, t])
            if denominator == 0.0:
                out[n, 0, t] = 0.0
                continue
            dop = np.sqrt(np.average(g1[n, :, t]) ** 2 + np.average(g2[n, :, t]) ** 2 + np.average(
                g3[n, :, t]) ** 2) / denominator
            out[n, 0, t] = np.clip(dop, 0.0, 1.0)
    return out


def splice_dop_in(data):
    dop = calculate_dop(data[:, 0], data[:, 1])
    data = np.abs(data).astype("float32")
    data[data == np.inf] = np.finfo(data.dtype).max
    new_train_x = np.zeros((data.shape[0], data.shape[1], data.shape[2] + 1,
                            data.shape[3]))
    new_train_x[:, :, :-1] = data
    new_train_x[:, :, -1, :] = dop
    data = new_train_x
    return data


class HeraPolarizationFullDataLoader(RawDataLoader):
    def _prepare_data(self):
        self.train_x[self.train_x == np.inf] = np.finfo(self.train_x.dtype).max
        self.test_x[self.test_x == np.inf] = np.finfo(self.test_x.dtype).max
        self.test_x = self.test_x.astype("float32")
        self.train_x = self.train_x.astype("float32")
        self.test_x = _normalize(self.test_x, self.test_y, 1, 4)
        self.train_x = _normalize(self.train_x, self.train_y, 1, 4)
        self.convert_pytorch()
        self.test_y = extract_polarization(self.test_y, 0)
        self.train_y = extract_polarization(self.train_y, 0)
        self.val_x = self.test_x.copy()
        self.val_y = self.test_y.copy()
        self.limit_datasets()

    def load_data(self, excluded_rfi: Union[str, None] = None):
        if excluded_rfi is None:
            rfi_models = []
            file_path = os.path.join(self.data_dir, "HERA-21-11-2024_all.pkl")
            data, _, masks = np.load(file_path, allow_pickle=True)
            train_x, train_y, test_x, test_y = test_train_split(data, masks)
        else:
            raise NotImplementedError("Polarization data loader does not support RFI exclusion")
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

class HeraPolarizationDeltaNormFullDataLoader(RawDataLoader):
    def _prepare_data(self):
        self.val_x = self.test_x.copy()
        self.val_y = self.test_y.copy()
        self.limit_datasets()

    def load_data(self, excluded_rfi: Union[str, None] = None):
        if excluded_rfi is None:
            rfi_models = []
            file_path = os.path.join(self.data_dir, "HERA-21-11-2024_all_delta_norm.pkl")
            data, masks = np.load(file_path, allow_pickle=True)
            train_x, train_y, test_x, test_y = test_train_split(data, masks)
        else:
            raise NotImplementedError("Polarization data loader does not support RFI exclusion")
        self.train_x = train_x
        self.train_y = train_y
        self.train_y = extract_polarization(self.train_y, 0)
        self.test_x = test_x
        self.test_y = test_y
        self.test_y = extract_polarization(self.test_y, 0)
        self.rfi_models = rfi_models
        self.original_size = self.train_x.shape[-1]
        self._prepare_data()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()


class HeraPolarizationDoPDataLoader(RawDataLoader):
    def _prepare_data(self):
        self.convert_pytorch()
        self.val_x = self.test_x.copy()
        self.val_y = self.test_y.copy()
        self.limit_datasets()

    def _normalize_data(self):
        self.test_x[:, :, :-1, :] = _normalize(self.test_x[:, :, :-1, :],
                                               self.test_y.astype("bool"), 1, 4)
        self.train_x[:, :, :-1, :] = _normalize(self.train_x[:, :, :-1, :],
                                                self.train_y.astype("bool"), 1, 4)
        self.val_x[:, :, :-1, :] = _normalize(self.val_x[:, :, :-1, :], self.val_y.astype("bool"),
                                              1, 4)

    def _convert_abs(self):
        self.test_x = np.abs(self.test_x).astype("float32")
        self.train_x = np.abs(self.train_x).astype("float32")
        self.val_x = np.abs(self.val_x).astype("float32")

    def _calc_dop(self):
        self.train_x = splice_dop_in(self.train_x)
        self.test_x = splice_dop_in(self.test_x)
        self.val_x = splice_dop_in(self.val_x)

    def load_data(self, excluded_rfi: Union[str, None] = None):
        if excluded_rfi is None:
            rfi_models = []
            file_path = os.path.join(self.data_dir, "HERA-25-11-2024_all.pkl")
            data, _, masks = np.load(file_path, allow_pickle=True)
            train_x, train_y, test_x, test_y = test_train_split(data, masks)
        else:
            raise NotImplementedError("Polarization data loader does not support RFI exclusion")
        self.train_x = np.moveaxis(train_x, 1, 2)
        self.train_y = np.moveaxis(train_y, 1, 2)
        self.test_x = np.moveaxis(test_x, 1, 2)
        self.test_y = np.moveaxis(test_y, 1, 2)
        self.rfi_models = rfi_models
        self.original_size = self.train_x.shape[1]
        self._prepare_data()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self._calc_dop()
        self._convert_abs()
        self._normalize_data()
        # Now extract polarization
        self.train_y = extract_polarization(self.train_y, 0)
        self.test_y = extract_polarization(self.test_y, 0)
        self.val_y = extract_polarization(self.val_y, 0)
        self.train_x = extract_polarization(self.train_x, 0)
        self.test_x = extract_polarization(self.test_x, 0)
        self.val_x = extract_polarization(self.val_x, 0)
        self.filter_noiseless_val_patches()
        self.filter_noiseless_train_patches()


class HeraPolarizationDeltaNormDoPDataLoader(RawDataLoader):
    def _prepare_data(self):
        self.val_x = self.test_x.copy()
        self.val_y = self.test_y.copy()
        self.limit_datasets()

    def _convert_abs(self):
        self.test_x = np.abs(self.test_x).astype("float32")
        self.train_x = np.abs(self.train_x).astype("float32")
        self.val_x = np.abs(self.val_x).astype("float32")

    def _calc_dop(self):
        self.train_x = splice_dop_in(self.train_x)
        self.test_x = splice_dop_in(self.test_x)
        self.val_x = splice_dop_in(self.val_x)

    def load_data(self, excluded_rfi: Union[str, None] = None):
        if excluded_rfi is None:
            rfi_models = []
            file_path = os.path.join(self.data_dir, "HERA-25-11-2024_all_delta_norm.pkl")
            data, masks = np.load(file_path, allow_pickle=True)
            train_x, train_y, test_x, test_y = test_train_split(data, masks)
        else:
            raise NotImplementedError("Polarization data loader does not support RFI exclusion")
        self.train_x = train_x
        self.train_y = train_y.astype("float32")
        self.test_x = test_x
        self.test_y = test_y.astype("float32")
        self.rfi_models = rfi_models
        self.original_size = self.train_x.shape[-1]
        self._prepare_data()
        if self.patch_size:
            self.create_patches(self.patch_size, self.stride)
        self._calc_dop()
        self._convert_abs()
        # Now extract polarization
        self.train_y = extract_polarization(self.train_y, 0)
        self.test_y = extract_polarization(self.test_y, 0)
        self.val_y = extract_polarization(self.val_y, 0)
        self.train_x = extract_polarization(self.train_x, 0)
        self.test_x = extract_polarization(self.test_x, 0)
        self.val_x = extract_polarization(self.val_x, 0)
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


class MixedDeltaNormLoader(RawDataLoader):

    def custom_limit_dataset(self, data, limit):
        if limit >= 1.0:
            return data
        int_limit = calc_limit_int(limit, len(data))
        return data[:int_limit]


    def load_data(self):
        file_path = os.path.join(self.data_dir, "LOFAR_Full_RFI_dataset_delta_norm.pkl")
        train_x, train_y, test_x, test_y, val_x, val_y = np.load(
            file_path, allow_pickle=True
        )
        file_path = os.path.join(self.data_dir, "HERA-21-11-2024_all_delta_norm.pkl")
        data, masks = np.load(
            file_path, allow_pickle=True
        )
        data = np.expand_dims(data[:, 0, :, :], 1)
        masks = np.expand_dims(masks[:, 0, :, :], 1)
        hera_train_x, hera_train_y, hera_test_x, hera_test_y = test_train_split(data, masks)

        self.test_x = _normalize(test_x, test_y, 3, 95)
        self.train_x = _normalize(train_x, train_y, 3, 95)
        hera_test_x = _normalize(hera_test_x, hera_test_y, 3, 95)
        hera_test_y = _normalize(hera_test_y, hera_test_y, 3, 95)
        self.train_x = np.concatenate((self.custom_limit_dataset(train_x, self.limit), self.custom_limit_dataset(hera_train_x, self.limit * 10)), axis=0)
        self.train_y = np.concatenate((self.custom_limit_dataset(train_y, self.limit), self.custom_limit_dataset(hera_train_y, self.limit * 10)), axis=0)
        self.test_x = np.concatenate((self.custom_limit_dataset(test_x, self.limit), self.custom_limit_dataset(hera_test_x, self.limit * 10)), axis=0)
        self.test_y = np.concatenate((self.custom_limit_dataset(test_y, self.limit), self.custom_limit_dataset(hera_test_y, self.limit * 10)), axis=0)
        self.val_x = val_x
        self.val_y = val_y
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


def create_delta_normalized_new_hera():
    file_path = os.path.join("data", "HERA-21-11-2024_all.pkl")
    data, _, masks = np.load(file_path, allow_pickle=True)
    train_x = np.moveaxis(data, 1, 2)
    train_y = np.moveaxis(masks, 1, 2)
    train_x[train_x == np.inf] = np.finfo(train_x.dtype).max
    train_x = train_x.astype("float32")
    train_x = _normalize(train_x, train_y, 1, 4)
    train_x = _delta_normalize(train_x)
    train_x = np.moveaxis(train_x, -1, 1).astype(np.float32)
    train_y = np.moveaxis(train_y, -1, 1).astype(np.float32)
    file_path = os.path.join("data", "HERA-21-11-2024_all_delta_norm.pkl")
    with open(file_path, "wb") as ofile:
        pickle.dump([train_x, train_y], ofile)


def create_delta_normalized_complex_hera():
    file_path = os.path.join("data", "HERA-25-11-2024_all.pkl")
    data, _, masks = np.load(file_path, allow_pickle=True)
    train_x = np.moveaxis(data, 1, 2)
    train_y = np.moveaxis(masks, 1, 2)
    train_x[train_x == np.inf] = np.finfo(train_x.dtype).max
    train_x = _normalize(train_x, train_y, 1, 4)
    train_x = _delta_normalize(train_x)
    train_x = np.moveaxis(train_x, -1, 1)
    train_y = np.moveaxis(train_y, -1, 1)
    file_path = os.path.join("data", "HERA-25-11-2024_all_delta_norm.pkl")
    with open(file_path, "wb") as ofile:
        pickle.dump([train_x, train_y], ofile)


def main():
    # create_delta_normalized_hera()
    # create_delta_normalized_lofar()
    # create_delta_normalized_new_hera()
    create_delta_normalized_complex_hera()


if __name__ == "__main__":
    main()
