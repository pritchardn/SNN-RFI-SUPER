"""
This module is responsible for building a configured data module.
"""

import torch
from torch.utils.data import TensorDataset
from data.data_module import ConfiguredDataModule
from interfaces.data.raw_data_loader import RawDataLoader
from interfaces.data.spiking_data_module import SpikeConverter


class DataModuleBuilder:
    def __init__(self):
        self.spike_encoder = None
        self.data_loader = None

    def reset(self):
        del self.data_loader
        del self.spike_encoder
        self.spike_encoder = None
        self.data_loader = None

    def set_dataset(self, data_loader: RawDataLoader):
        self.data_loader = data_loader

    def set_encoding(self, spike_encoder: SpikeConverter):
        self.spike_encoder = spike_encoder

    """
    def add_transforms(self, transforms: list):
        # TODO: Add transforms later
        pass
    """

    def build(self, batch_size: int) -> ConfiguredDataModule:
        if self.data_loader and self.spike_encoder:
            self.data_loader.load_data()
            train_x = torch.from_numpy(
                self.spike_encoder.encode_x(self.data_loader.fetch_train_x())
            )
            train_y = torch.from_numpy(
                self.spike_encoder.encode_y(self.data_loader.fetch_train_y())
            )
            test_x = torch.from_numpy(
                self.spike_encoder.encode_x(self.data_loader.fetch_test_x())
            )
            test_y = torch.from_numpy(
                self.data_loader.fetch_test_y()  # Original labels.
            )
            val_x = torch.from_numpy(
                self.spike_encoder.encode_x(self.data_loader.fetch_val_x())
            )
            val_y = torch.from_numpy(
                self.spike_encoder.encode_y(self.data_loader.fetch_val_y())
            )
            train_dset = TensorDataset(train_x, train_y)
            test_dset = TensorDataset(test_x, test_y)
            val_dset = TensorDataset(val_x, val_y)
            self.data_loader.release_memory()
            return ConfiguredDataModule(train_dset, test_dset, val_dset, batch_size, self.data_loader.stride)
        raise Exception("DataModuleBuilder has not been properly configured.")
