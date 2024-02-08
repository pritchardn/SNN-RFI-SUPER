import glob
import json
import os

import pytorch_lightning as pl

from data.data_loaders import HeraDataLoader
from data.data_module import ConfiguredDataModule
from data.data_module_builder import DataModuleBuilder
from data.spike_converters import LatencySpikeConverter
from data.utils import reconstruct_patches
from evaluation import final_evaluation
from interfaces.data.raw_data_loader import RawDataLoader
from interfaces.data.spiking_data_module import SpikeConverter
from models.fc_latency import LitFcLatency


def data_source_from_config(config: dict) -> RawDataLoader:
    data_path = config.get("data_path")
    patch_size = config.get("patch_size")
    stride = config.get("stride")
    limit = config.get("limit")
    dataset = config.get("dataset")
    if dataset == "HERA":
        data_source = HeraDataLoader(
            data_path, patch_size=patch_size, stride=stride, limit=limit
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported.")
    return data_source


def dataset_from_config(
    config: dict, data_source: RawDataLoader, encoder: SpikeConverter
) -> ConfiguredDataModule:
    batch_size = config.get("batch_size")
    data_builder = DataModuleBuilder()
    data_builder.set_dataset(data_source)
    data_builder.set_encoding(encoder)
    dataset = data_builder.build(batch_size)
    return dataset


def model_from_config(config: dict) -> pl.LightningModule:
    model_type = config.get("type")
    beta = config.get("beta")
    if model_type == "FC_LATENCY":
        num_inputs = config.get("num_inputs")
        num_hidden = config.get("num_hidden")
        num_outputs = config.get("num_outputs")
        model = LitFcLatency(num_inputs, num_hidden, num_outputs, beta)
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")
    return model


def trainer_from_config(config: dict, root_dir: str) -> pl.Trainer:
    epochs = config.get("epochs")
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, min_delta=1e-4
    )
    trainer = pl.trainer.Trainer(
        max_epochs=epochs,
        benchmark=True,
        callbacks=[early_stopping_callback],
        default_root_dir=root_dir,
    )
    return trainer


def encoder_from_config(config: dict) -> SpikeConverter:
    encoder = None
    if config.get("method") == "LATENCY":
        exposure = config.get("exposure")
        tau = config.get("tau")
        normalize = config.get("normalize")
        encoder = LatencySpikeConverter(exposure=exposure, tau=tau, normalize=normalize)
    return encoder


class Experiment:
    def __init__(self, root_dir="./"):
        self.data_source = None
        self.dataset = None
        self.model = None
        self.trainer = None
        self.encoder = None
        self.configuration = None
        self.ready = False
        self.checkpoint_path = None
        self.root_dir = root_dir

    def from_config(self, config: dict):
        self.configuration = config
        if self.configuration.get("encoder"):
            self.encoder = encoder_from_config(config.get("encoder"))
        if self.configuration.get("data_source"):
            self.data_source = data_source_from_config(config.get("data_source"))
            if self.configuration.get("dataset") and self.encoder:
                self.dataset = dataset_from_config(
                    config.get("dataset"), self.data_source, self.encoder
                )
        if self.configuration.get("model"):
            self.model = model_from_config(config.get("model"))
        if self.configuration.get("trainer"):
            self.trainer = trainer_from_config(config.get("trainer"), self.root_dir)

    def from_checkpoint(self, working_dir: str):
        self.load_config(os.path.join(working_dir, "config.json"))
        checkpoint_dir = os.path.join(working_dir, "checkpoints", "*.ckpt")
        checkpoint_file = glob.glob(checkpoint_dir)[0]
        self.checkpoint_path = checkpoint_file
        self.from_config(self.configuration)

    def add_dataset(self, data_source: RawDataLoader):
        self.data_source = data_source
        self.ready = False

    def save_config(self):
        out_dir = self.trainer.log_dir
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "config.json"), "w") as ofile:
            json.dump(self.configuration, ofile, indent=4)

    def load_config(self, config_path: str):
        with open(config_path, "r") as ifile:
            self.configuration = json.load(ifile)

    def prepare(self):
        err_msg = ""
        if not self.ready:
            if self.data_source and self.encoder:
                self.dataset = dataset_from_config(
                    self.configuration.get("dataset"), self.data_source, self.encoder
                )
            else:
                err_msg += "Data source not set.\n"
            if not self.model:
                err_msg += "Model not set.\n"
            if not self.trainer:
                err_msg += "Trainer not set.\n"
            else:
                self.save_config()
            if not self.encoder:
                err_msg += "Encoder not set.\n"
            if err_msg != "":
                raise ValueError(err_msg)
            else:
                self.ready = True

    def train(self):
        if not self.ready:
            raise RuntimeError("Experiment not ready.")
        self.model.train()
        if self.checkpoint_path:
            self.trainer.fit(self.model, self.dataset, ckpt_path=self.checkpoint_path)
        else:
            self.trainer.fit(self.model, self.dataset)

    def evaluate(self):
        self.model.eval()
        mask_orig = reconstruct_patches(
            self.data_source.fetch_test_y(),
            self.data_source.original_size,
            self.data_source.stride,
        )
        metrics = final_evaluation(
            self.model, self.dataset, self.encoder, mask_orig, self.trainer.log_dir
        )
        return metrics