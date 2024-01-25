from data.data_loaders import HeraDataLoader
from data.data_module_builder import DataModuleBuilder
from data.spike_encoders import LatencySpikeEncoder


def main():
    data_builder = DataModuleBuilder()
    data_source = HeraDataLoader("./data")
    data_builder.set_dataset(data_source)
    spike_converter = LatencySpikeEncoder(exposure=32, tau=1.0, normalize=True)
    data_builder.set_encoding(spike_converter)
    data_module = data_builder.build(32)


if __name__ == "__main__":
    main()
