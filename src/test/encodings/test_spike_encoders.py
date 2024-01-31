from unittest import TestCase

import numpy as np

from data.data_loaders import HeraDataLoader
from data.data_module_builder import DataModuleBuilder
from data.spike_converters import LatencySpikeConverter
from interfaces.data.raw_data_loader import RawDataLoader


def _fetch_example_dataset():
    data_builder = DataModuleBuilder()
    data_source = HeraDataLoader("./data", limit=0.1, patch_size=32, stride=32)
    data_builder.set_dataset(data_source)
    return data_source, data_builder


def _test_decode_inference(data_source: RawDataLoader, data_builder: DataModuleBuilder,
                           exposure: int):
    converter = LatencySpikeConverter(exposure=exposure, tau=1.0, normalize=True)
    data_builder.set_encoding(converter)
    data_builder.build(32)
    mask = data_source.test_y
    z = converter.decode_inference(converter.encode_y(mask))
    return np.equal(mask, z).all()


class TestLatencySpikeEncoder(TestCase):

    def test_decode_inference_1(self):
        data_source, data_builder = _fetch_example_dataset()
        self.assertFalse(_test_decode_inference(data_source, data_builder, 1))

    def test_decode_inference_2(self):
        data_source, data_builder = _fetch_example_dataset()
        self.assertTrue(_test_decode_inference(data_source, data_builder, 2))

    def test_decode_inference_4(self):
        data_source, data_builder = _fetch_example_dataset()
        self.assertTrue(_test_decode_inference(data_source, data_builder, 4))

    def test_decode_inference_16(self):
        data_source, data_builder = _fetch_example_dataset()
        self.assertTrue(_test_decode_inference(data_source, data_builder, 16))
