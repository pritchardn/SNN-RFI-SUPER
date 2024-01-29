import pytorch_lightning as pl

from data.data_loaders import HeraDataLoader
from data.data_module_builder import DataModuleBuilder
from data.spike_encoders import LatencySpikeEncoder
from models.fc_latency import LitFcLatency


def main():
    EXPOSURE = 32
    TAU = 1.0
    BETA = 0.95
    data_builder = DataModuleBuilder()
    data_source = HeraDataLoader("./data", limit=0.1, patch_size=32, stride=32)
    data_builder.set_dataset(data_source)
    spike_converter = LatencySpikeEncoder(exposure=EXPOSURE, tau=TAU, normalize=True)
    data_builder.set_encoding(spike_converter)
    data_module = data_builder.build(32)
    print("Built data module")
    model = LitFcLatency(32, 128, 32, EXPOSURE, BETA)
    print("Built model")
    trainer = pl.trainer.Trainer()
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
