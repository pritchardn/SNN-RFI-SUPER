import glob
import json
import os

import pytorch_lightning as pl
import torch

from data.data_loaders import HeraDataLoader, LofarDataLoader, TabascalDataLoader
from data.data_module import ConfiguredDataModule
from data.data_module_builder import DataModuleBuilder
from data.spike_converters import (
    LatencySpikeConverter,
    RateSpikeConverter,
    DeltaSpikeConverter,
)
from data.spike_converters.ForwardStepConverter import ForwardStepConverter
from interfaces.data.raw_data_loader import RawDataLoader
from interfaces.data.spiking_data_module import SpikeConverter
from models.fc_delta import LitFcDelta
from models.fc_forwardstep import LitFcForwardStep
from models.fc_latency import LitFcLatency
from models.fc_rate import LitFcRate


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
    elif dataset == "LOFAR":
        data_source = LofarDataLoader(
            data_path, patch_size=patch_size, stride=stride, limit=limit
        )
    elif dataset == "TABASCAL":
        data_source = TabascalDataLoader(
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
    elif model_type == "FC_RATE":
        num_inputs = config.get("num_inputs")
        num_hidden = config.get("num_hidden")
        num_outputs = config.get("num_outputs")
        model = LitFcRate(num_inputs, num_hidden, num_outputs, beta)
    elif model_type == "FC_DELTA":
        num_inputs = config.get("num_inputs")
        num_hidden = config.get("num_hidden")
        num_outputs = config.get("num_outputs")
        reconstruct_loss = config.get("reconstruct_loss")
        model = LitFcDelta(
            num_inputs, num_hidden, num_outputs, beta, reconstruct_loss, True
        )
    elif model_type == "FC_DELTA_ON":
        num_inputs = config.get("num_inputs")
        num_hidden = config.get("num_hidden")
        num_outputs = config.get("num_outputs")
        reconstruct_loss = config.get("reconstruct_loss")
        model = LitFcDelta(
            num_inputs, num_hidden, num_outputs, beta, reconstruct_loss, False
        )
    elif model_type == "FC_FORWARD_STEP":
        num_inputs = config.get("num_inputs")
        num_hidden = config.get("num_hidden")
        num_outputs = config.get("num_outputs")
        model = LitFcForwardStep(num_inputs, num_hidden, num_outputs, beta)
    else:
        raise NotImplementedError(f"Model type {model_type} is not supported.")
    return model


def trainer_from_config(config: dict, root_dir: str) -> pl.Trainer:
    num_gpus = torch.cuda.device_count()
    epochs = config.get("epochs")
    patience = config.get("patience", 10)
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=patience, min_delta=1e-4
    )
    if num_gpus > 0:
        trainer = pl.trainer.Trainer(
            max_epochs=epochs,
            benchmark=True,
            callbacks=[early_stopping_callback],
            default_root_dir=root_dir,
            devices=num_gpus,
            num_nodes=config.get("num_nodes"),
        )
    else:
        trainer = pl.trainer.Trainer(
            max_epochs=epochs,
            benchmark=True,
            callbacks=[early_stopping_callback],
            default_root_dir=root_dir,
            num_nodes=config.get("num_nodes"),
        )
    return trainer


def encoder_from_config(config: dict) -> SpikeConverter:
    encoder = None
    if config.get("method") == "LATENCY":
        exposure = config.get("exposure")
        tau = config.get("tau")
        normalize = config.get("normalize")
        encoder = LatencySpikeConverter(exposure=exposure, tau=tau, normalize=normalize)
    elif config.get("method") == "RATE":
        exposure = config.get("exposure")
        encoder = RateSpikeConverter(exposure=exposure)
    elif config.get("method") == "DELTA":
        threshold = config.get("threshold")
        off_spikes = config.get("off_spikes")
        encoder = DeltaSpikeConverter(threshold=threshold, off_spikes=off_spikes)
    elif config.get("method") == "FORWARDSTEP":
        threshold = config.get("threshold")
        exposure = config.get("exposure")
        tau = config.get("tau")
        normalize = config.get("normalize")
        exposure_mode = config.get("exposure_mode")
        encoder = ForwardStepConverter(
            threshold=threshold,
            exposure=exposure,
            tau=tau,
            normalize=normalize,
            exposure_mode=exposure_mode,
        )
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
        self.dataset = None
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
                if not self.dataset:
                    self.dataset = dataset_from_config(
                        self.configuration.get("dataset"),
                        self.data_source,
                        self.encoder,
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
            else:
                self.model.set_converter(self.encoder)
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
        metrics = self.trainer.test(self.model, self.dataset.test_dataloader())
        accuracy = metrics[0]["test_accuracy"]
        mse = metrics[0]["test_mse"]
        auroc = metrics[0]["test_auroc"]
        auprc = metrics[0]["test_auprc"]
        f1 = metrics[0]["test_f1"]
        output = json.dumps(
            {
                "accuracy": accuracy,
                "mse": mse,
                "auroc": auroc,
                "auprc": auprc,
                "f1": f1,
            }
        )
        # Write output
        with open(os.path.join(self.trainer.log_dir, "metrics.json"), "w") as ofile:
            json.dump(output, ofile, indent=4)
        return accuracy, mse, auroc, auprc, f1
