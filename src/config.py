"""
This module contains the default configuration parameters for the different models.
"""

import copy
import os

DEFAULT_HERA_LATENCY = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 36,
    },
    "model": {
        "type": "FC_LATENCY",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "num_layers": 6,
        "beta": 0.245507490258551,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "LATENCY",
        "exposure": 10,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_HERA_LATENCY_RNN = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_HERA_LATENCY_RNN["model"]["type"] = "RNN_LATENCY"
DEFAULT_LOFAR_LATENCY = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_LOFAR_LATENCY["data_source"]["dataset"] = "LOFAR"
DEFAULT_LOFAR_LATENCY["model"]["beta"] = 0.148676888253826
DEFAULT_LOFAR_LATENCY["model"]["num_hidden"] = 512
DEFAULT_LOFAR_LATENCY["model"]["num_layers"] = 2
DEFAULT_LOFAR_LATENCY["encoder"]["exposure"] = 61

DEFAULT_HERA_LATENCY_MH = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 512,
        "stride": 512,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 4,
    },
    "model": {
        "type": "MH_LATENCY",
        "num_inputs": 512,
        "num_hidden": 2048,
        "num_outputs": 512,
        "num_hidden_layers": 2,
        "alpha": 0.10,
        "beta": 0.245507490258551,
        "head_width": 16,
        "head_stride": 16,
        "learning_rate": 1e-3,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "LATENCY",
        "exposure": 10,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_HERA_LATENCY_DIVNORM = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_HERA_LATENCY_DIVNORM["model"]["num_hidden"] = 256
DEFAULT_HERA_LATENCY_DIVNORM["model"]["beta"] = 0.239039586173328
DEFAULT_HERA_LATENCY_DIVNORM["model"]["num_layers"] = 5
DEFAULT_HERA_LATENCY_DIVNORM["data_source"]["delta_normalization"] = True
DEFAULT_HERA_LATENCY_DIVNORM["encoder"]["exposure"] = 4

DEFAULT_LOFAR_LATENCY_DIVNORM = copy.deepcopy(DEFAULT_LOFAR_LATENCY)
DEFAULT_LOFAR_LATENCY_DIVNORM["model"]["num_hidden"] = 256
DEFAULT_LOFAR_LATENCY_DIVNORM["model"]["beta"] = 0.675127571586639
DEFAULT_LOFAR_LATENCY_DIVNORM["model"]["num_layers"] = 2
DEFAULT_LOFAR_LATENCY_DIVNORM["data_source"]["delta_normalization"] = True
DEFAULT_LOFAR_LATENCY_DIVNORM["encoder"]["exposure"] = 62

DEFAULT_HERA_RATE = {
    "data_source": {
        "data_path": "./data",
        "limit": 0.1,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 107,
    },
    "model": {
        "type": "FC_RATE",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "num_layers": 2,
        "beta": 0.599215428763344,
    },
    "trainer": {
        "epochs": 50,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "RATE",
        "exposure": 4,
        "tau": 1.0,
        "normalize": True,
    },
}
DEFAULT_HERA_RATE_RNN = copy.deepcopy(DEFAULT_HERA_RATE)
DEFAULT_HERA_RATE_RNN["model"]["type"] = "RNN_RATE"
DEFAULT_LOFAR_RATE = copy.deepcopy(DEFAULT_HERA_RATE)
DEFAULT_LOFAR_RATE["data_source"]["dataset"] = "LOFAR"

DEFAULT_HERA_DELTA = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 48,
    },
    "model": {
        "type": "FC_DELTA",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 64,
        "num_layers": 2,
        "beta": 0.504376656494665,
        "reconstruct_loss": False,
    },
    "trainer": {
        "epochs": 80,
        "num_nodes": int(os.getenv("NNODES", 1)),
        "patience": 100,
    },
    "encoder": {"method": "DELTA", "threshold": 0.1, "off_spikes": True},
}

DEFAULT_HERA_DELTA_RNN = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_HERA_DELTA_RNN["model"]["type"] = "RNN_DELTA"
DEFAULT_LOFAR_DELTA = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_LOFAR_DELTA["data_source"]["dataset"] = "LOFAR"

DEFAULT_HERA_DELTA_ON = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 36,
    },
    "model": {
        "type": "FC_DELTA_ON",
        "num_inputs": 64,
        "num_hidden": 128,
        "num_outputs": 64,
        "num_layers": 2,
        "beta": 0.7270826938643781,
        "reconstruct_loss": False,
    },
    "trainer": {
        "epochs": 50,
        "num_nodes": int(os.getenv("NNODES", 1)),
        "patience": 100,
    },
    "encoder": {"method": "DELTA", "threshold": 0.1, "off_spikes": False},
}

DEFAULT_HERA_DELTA_ON_RNN = copy.deepcopy(DEFAULT_HERA_DELTA_ON)
DEFAULT_HERA_DELTA_ON_RNN["model"]["type"] = "RNN_DELTA_ON"
DEFAULT_LOFAR_DELTA_ON = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_LOFAR_DELTA_ON["data_source"]["dataset"] = "LOFAR"

DEFAULT_HERA_DELTA_EXPOSURE = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 36,
    },
    "model": {
        "type": "FC_DELTA_EXPOSURE",
        "num_inputs": 32,
        "num_hidden": 256,
        "num_outputs": 32,
        "num_layers": 5,
        "beta": 0.70365295492966,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {"method": "DELTA_EXPOSURE", "threshold": 0.1, "exposure": 35},
}

DEFAULT_HERA_DELTA_EXPOSURE_RNN = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE)
DEFAULT_HERA_DELTA_EXPOSURE_RNN["model"]["type"] = "RNN_DELTA"
DEFAULT_LOFAR_DELTA_EXPOSURE = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE)
DEFAULT_LOFAR_DELTA_EXPOSURE["data_source"]["dataset"] = "LOFAR"
DEFAULT_LOFAR_DELTA_EXPOSURE["encoder"]["threshold"] = 0.5
DEFAULT_LOFAR_DELTA_EXPOSURE["model"]["num_hidden"] = 512
DEFAULT_LOFAR_DELTA_EXPOSURE["model"]["num_layers"] = 5
DEFAULT_LOFAR_DELTA_EXPOSURE["model"]["beta"] = 0.374296935613819
DEFAULT_LOFAR_DELTA_EXPOSURE["encoder"]["exposure"] = 62

DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE)
DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM["model"]["num_hidden"] = 512
DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM["model"]["beta"] = 0.524985471120514
DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM["model"]["num_layers"] = 4
DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM["data_source"]["delta_normalization"] = True
DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM["encoder"]["exposure"] = 21

DEFAULT_LOFAR_DELTA_EXPOSURE_DIVNORM = copy.deepcopy(DEFAULT_LOFAR_DELTA_EXPOSURE)
DEFAULT_LOFAR_DELTA_EXPOSURE_DIVNORM["model"]["num_hidden"] = 512
DEFAULT_LOFAR_DELTA_EXPOSURE_DIVNORM["model"]["num_layers"] = 2
DEFAULT_LOFAR_DELTA_EXPOSURE_DIVNORM["model"]["beta"] = 0.734944135280315
DEFAULT_LOFAR_DELTA_EXPOSURE_DIVNORM["data_source"]["delta_normalization"] = True
DEFAULT_LOFAR_DELTA_EXPOSURE_DIVNORM["encoder"]["exposure"] = 55

DEFAULT_HERA_FORWARD = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 36,
    },
    "model": {
        "type": "FC_FORWARD_STEP",
        "num_inputs": 64,
        "num_hidden": 512,
        "num_outputs": 32,
        "num_layers": 6,
        "beta": 0.10223713665629142,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "FORWARDSTEP",
        "exposure": 9,
        "tau": 1.0,
        "threshold": 0.1,
        "normalize": True,
        "exposure_mode": "direct",
    },
}

DEFAULT_HERA_FORWARD_RNN = copy.deepcopy(DEFAULT_HERA_FORWARD)
DEFAULT_HERA_FORWARD_RNN["model"]["type"] = "RNN_FORWARD_STEP"
DEFAULT_LOFAR_FORWARD = copy.deepcopy(DEFAULT_HERA_FORWARD)
DEFAULT_LOFAR_FORWARD["data_source"]["dataset"] = "LOFAR"
DEFAULT_LOFAR_FORWARD["model"]["num_hidden"] = 512
DEFAULT_LOFAR_FORWARD["model"]["beta"] = 0.511618348677156
DEFAULT_LOFAR_FORWARD["model"]["num_layers"] = 2
DEFAULT_LOFAR_FORWARD["encoder"]["exposure"] = 24
DEFAULT_LOFAR_FORWARD["encoder"]["exposure_mode"] = "direct"

DEFAULT_HERA_FORWARD_DIVNORM = copy.deepcopy(DEFAULT_HERA_FORWARD)
DEFAULT_HERA_FORWARD_DIVNORM["model"]["num_hidden"] = 256
DEFAULT_HERA_FORWARD_DIVNORM["model"]["beta"] = 0.0254787188709035
DEFAULT_HERA_FORWARD_DIVNORM["model"]["num_layers"] = 5
DEFAULT_HERA_FORWARD_DIVNORM["data_source"]["delta_normalization"] = True
DEFAULT_HERA_FORWARD_DIVNORM["encoder"]["exposure"] = 33
DEFAULT_HERA_FORWARD_DIVNORM["encoder"]["exposure_mode"] = "direct"

DEFAULT_LOFAR_FORWARD_DIVNORM = copy.deepcopy(DEFAULT_LOFAR_FORWARD)
DEFAULT_LOFAR_FORWARD_DIVNORM["model"]["num_hidden"] = 512
DEFAULT_LOFAR_FORWARD_DIVNORM["model"]["beta"] = 0.37324889843381
DEFAULT_LOFAR_FORWARD_DIVNORM["model"]["num_layers"] = 4
DEFAULT_LOFAR_FORWARD_DIVNORM["data_source"]["delta_normalization"] = True
DEFAULT_LOFAR_FORWARD_DIVNORM["encoder"]["exposure"] = 30
DEFAULT_LOFAR_FORWARD_DIVNORM["encoder"]["exposure_mode"] = "direct"

DEFAULT_HERA_ANN = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 36,
    },
    "model": {
        "type": "FC_ANN",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "num_layers": 2,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "ANN",
    },
}

DEFAULT_HERA_ANN_DIVNORM = copy.deepcopy(DEFAULT_HERA_ANN)
DEFAULT_HERA_ANN_DIVNORM["model"]["num_hidden"] = 512
DEFAULT_HERA_ANN_DIVNORM["model"]["num_layers"] = 3
DEFAULT_HERA_ANN_DIVNORM["data_source"]["delta_normalization"] = True

DEFAULT_LOFAR_ANN = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "LOFAR",
    },
    "dataset": {
        "batch_size": 36,
    },
    "model": {
        "type": "FC_ANN",
        "num_inputs": 32,
        "num_hidden": 512,
        "num_outputs": 32,
        "num_layers": 6,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "ANN",
    },
}

DEFAULT_LOFAR_ANN_DIVNORM = copy.deepcopy(DEFAULT_LOFAR_ANN)
DEFAULT_LOFAR_ANN_DIVNORM["model"]["num_hidden"] = 512
DEFAULT_LOFAR_ANN_DIVNORM["model"]["num_layers"] = 6
DEFAULT_LOFAR_ANN_DIVNORM["data_source"]["delta_normalization"] = True


def get_default_params(
        dataset: str,
        model_type: str,
        model_size: int = 128,
        exposure_mode: str = None,
        delta_normalization: bool = False,
):
    if dataset == "HERA":
        if model_type == "FC_LATENCY":
            if delta_normalization:
                params = DEFAULT_HERA_LATENCY_DIVNORM
            else:
                params = DEFAULT_HERA_LATENCY
        elif model_type == "FC_LATENCY_XYLO":
            if delta_normalization:
                params = DEFAULT_HERA_LATENCY_DIVNORM
            else:
                params = DEFAULT_HERA_LATENCY
            params["model"]["num_inputs"] = 8
            params["model"]["num_outputs"] = 8
            params["model"]["num_layers"] = 3
            params["model"]["num_hidden"] = 128
            params["data_source"]["patch_size"] = 8
            params["data_source"]["stride"] = 8
        elif model_type == "RNN_LATENCY":
            params = DEFAULT_HERA_LATENCY_RNN
        elif model_type == "FC_RATE":
            params = DEFAULT_HERA_RATE
        elif model_type == "RNN_RATE":
            params = DEFAULT_HERA_RATE_RNN
        elif model_type == "FC_DELTA":
            params = DEFAULT_HERA_DELTA
        elif model_type == "RNN_DELTA":
            params = DEFAULT_HERA_DELTA_RNN
        elif model_type == "FC_DELTA_ON":
            params = DEFAULT_HERA_DELTA_ON
        elif model_type == "RNN_DELTA_ON":
            params = DEFAULT_HERA_DELTA_ON_RNN
        elif model_type == "FC_DELTA_EXPOSURE":
            if delta_normalization:
                params = DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM
            else:
                params = DEFAULT_HERA_DELTA_EXPOSURE
        elif model_type == "RNN_DELTA_EXPOSURE":
            params = DEFAULT_HERA_DELTA_EXPOSURE_RNN
        elif model_type == "FC_FORWARD_STEP":
            if exposure_mode == "direct":
                if delta_normalization:
                    params = copy.deepcopy(DEFAULT_HERA_FORWARD_DIVNORM)
                else:
                    params = copy.deepcopy(DEFAULT_HERA_FORWARD)
            elif exposure_mode == "latency":
                params = copy.deepcopy(DEFAULT_HERA_FORWARD)
                params["encoder"]["exposure_mode"] = "latency"
                params["encoder"]["exposure"] = 22
                params["trainer"]["epochs"] = 83
                params["model"]["beta"] = 0.920967991589638
                params["dataset"]["batch_size"] = 54
            elif exposure_mode == "first":
                params = copy.deepcopy(DEFAULT_HERA_FORWARD)
                params["encoder"]["exposure_mode"] = "first"
            else:
                params = DEFAULT_HERA_FORWARD
        elif model_type == "RNN_FORWARD_STEP":
            if exposure_mode == "direct":
                params = copy.deepcopy(DEFAULT_HERA_FORWARD_RNN)
                params["encoder"]["exposure_mode"] = "direct"
            elif exposure_mode == "latency":
                params = copy.deepcopy(DEFAULT_HERA_FORWARD_RNN)
                params["encoder"]["exposure_mode"] = "latency"
            else:
                params = DEFAULT_HERA_FORWARD_RNN
        elif model_type == "FC_ANN":
            if delta_normalization:
                params = DEFAULT_HERA_ANN_DIVNORM
            else:
                params = DEFAULT_HERA_ANN
        elif model_type == "FCP_ANN":
            params = copy.deepcopy(DEFAULT_HERA_ANN)
            params["model"]["num_hidden"] = model_size
            params["model"]["type"] = model_type
            stride = params["data_source"]["stride"]
            params["model"]["num_inputs"] = stride * stride
            params["model"]["num_outputs"] = stride * stride
            params["model"]["num_hidden"] = model_size
            params["encoder"]["method"] = "ANN_PATCHED"
        elif model_type == "FCP_LATENCY":
            params = DEFAULT_HERA_LATENCY
            stride = params["data_source"]["stride"]
            params["model"]["type"] = model_type
            params["model"]["num_inputs"] = stride * stride
            params["model"]["num_outputs"] = stride * stride
            params["model"]["num_hidden"] = model_size
        elif model_type == "FCP_RATE":
            params = DEFAULT_HERA_RATE
            stride = params["data_source"]["stride"]
            params["model"]["type"] = model_type
            params["model"]["num_inputs"] = stride * stride
            params["model"]["num_outputs"] = stride * stride
            params["model"]["num_hidden"] = model_size
            params["encoder"]["exposure"] = 64
            params["trainer"]["epochs"] = 100
        elif model_type == "FCP_DELTA":
            params = DEFAULT_HERA_DELTA
            stride = params["data_source"]["stride"]
            params["model"]["type"] = model_type
            params["model"]["num_inputs"] = stride * stride
            params["model"]["num_outputs"] = stride * stride * 2
            params["model"]["num_hidden"] = model_size
        elif model_type == "FCP_DELTA_ON":
            params = DEFAULT_HERA_DELTA_ON
            stride = params["data_source"]["stride"]
            params["model"]["type"] = model_type
            params["model"]["num_inputs"] = stride * stride * 2
            params["model"]["num_outputs"] = stride * stride * 2
            params["model"]["num_hidden"] = model_size
        elif model_type == "FCP_FORWARD_STEP":
            if exposure_mode == "direct":
                params = copy.deepcopy(DEFAULT_HERA_FORWARD)
                params["model"]["type"] = model_type
                stride = params["data_source"]["stride"]
                params["model"]["num_inputs"] = stride * stride * 2
                params["model"]["num_outputs"] = stride * stride
                params["model"]["num_hidden"] = model_size
                params["encoder"]["exposure_mode"] = "direct"
                params["encoder"]["exposure"] = 50
                params["trainer"]["epochs"] = 43
                params["model"]["beta"] = 0.920343975816805
                params["dataset"]["batch_size"] = 79
            elif exposure_mode == "latency":
                params = copy.deepcopy(DEFAULT_HERA_FORWARD)
                params["model"]["type"] = model_type
                stride = params["data_source"]["stride"]
                params["model"]["num_inputs"] = stride * stride * 2
                params["model"]["num_outputs"] = stride * stride
                params["model"]["num_hidden"] = model_size
                params["encoder"]["exposure_mode"] = "latency"
                params["encoder"]["exposure"] = 22
                params["trainer"]["epochs"] = 83
                params["model"]["beta"] = 0.920967991589638
                params["dataset"]["batch_size"] = 54
            else:
                params = copy.deepcopy(DEFAULT_HERA_FORWARD)
            stride = params["data_source"]["stride"]
            params["model"]["num_inputs"] = stride * stride * 2
            params["model"]["num_outputs"] = stride * stride
            params["model"]["num_hidden"] = model_size
            params["model"]["type"] = model_type
        elif model_type == "MH_LATENCY":
            params = copy.deepcopy(DEFAULT_HERA_LATENCY_MH)
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "HERA_POLAR_FULL":
        if model_type == "FC_LATENCY":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            params["data_source"]["dataset"] = dataset
            params["data_source"]["patch_size"] = 8
            params["data_source"]["stride"] = 8
            params["dataset"]["batch_size"] = params["dataset"]["batch_size"] * 4
        elif model_type == "FC_LATENCY_XYLO":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            params["data_source"]["dataset"] = dataset
            params["model"]["num_inputs"] = 16
            params["model"]["num_outputs"] = 4
            params["data_source"]["patch_size"] = 4
            params["data_source"]["stride"] = 4
            params["dataset"]["batch_size"] = params["dataset"]["batch_size"] * 8
        elif model_type == "FC_LATENCY_FULL":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            params["data_source"]["dataset"] = dataset
            params["data_source"]["patch_size"] = 512
            params["data_source"]["stride"] = 512
            params["model"]["num_inputs"] = 2048
            params["model"]["num_outputs"] = 2048
            params["model"]["num_hidden"] = 2048
            params["dataset"]["batch_size"] = params["dataset"]["batch_size"] // 8
        elif model_type == "FC_DELTA_EXPOSURE":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE)
            params["data_source"]["dataset"] = dataset
            params["data_source"]["patch_size"] = 8
            params["data_source"]["stride"] = 8
            params["dataset"]["batch_size"] = params["dataset"]["batch_size"] * 4
        elif model_type == "FC_DELTA_EXPOSURE_XYLO":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE)
            params["data_source"]["dataset"] = dataset
            params["model"]["num_inputs"] = 16
            params["model"]["num_outputs"] = 8
            params["data_source"]["patch_size"] = 4
            params["data_source"]["stride"] = 4
            params["dataset"]["batch_size"] = params["dataset"]["batch_size"] * 8
        else:
            raise NotImplementedError(f"No other model types have been tested for {dataset}")
    elif dataset == "HERA_POLAR_DOP":
        if model_type == "FC_LATENCY":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            params["data_source"]["dataset"] = dataset
            params["model"]["num_inputs"] = params["model"]["num_inputs"] + 1
        elif model_type == "FC_LATENCY_XYLO":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            params["data_source"]["dataset"] = dataset
            params["model"]["num_inputs"] = 9
            params["model"]["num_outputs"] = 8
            params["data_source"]["patch_size"] = 8
            params["data_source"]["stride"] = 8
            params["dataset"]["batch_size"] = params["dataset"]["batch_size"] * 2
        elif model_type == "FC_LATENCY_FULL":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
            params["data_source"]["dataset"] = dataset
            params["data_source"]["patch_size"] = 512
            params["data_source"]["stride"] = 512
            params["model"]["num_inputs"] = 513
            params["model"]["num_outputs"] = 512
            params["model"]["num_hidden"] = 2048
            params["dataset"]["batch_size"] = params["dataset"]["batch_size"] // 8
        elif model_type == "FC_DELTA_EXPOSURE":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE)
            params["data_source"]["dataset"] = dataset
            params["model"]["num_inputs"] = params["model"]["num_inputs"] + 1
        elif model_type == "FC_DELTA_EXPOSURE_XYLO":
            if delta_normalization:
                params = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE_DIVNORM)
            else:
                params = copy.deepcopy(DEFAULT_HERA_DELTA_EXPOSURE)
            params["data_source"]["dataset"] = dataset
            params["model"]["num_inputs"] = 9
            params["model"]["num_outputs"] = 8
            params["data_source"]["patch_size"] = 8
            params["data_source"]["stride"] = 8
            params["dataset"]["batch_size"] = params["dataset"]["batch_size"] * 2
        else:
            raise NotImplementedError(f"No other model types have been tested for {dataset}")
    elif dataset == "LOFAR":
        if model_type == "FC_LATENCY" or model_type == "RNN_LATENCY":
            if delta_normalization:
                params = DEFAULT_LOFAR_LATENCY_DIVNORM
            else:
                params = DEFAULT_LOFAR_LATENCY
        elif model_type == "FC_RATE" or model_type == "RNN_RATE":
            params = DEFAULT_LOFAR_RATE
        elif model_type == "FC_DELTA" or model_type == "RNN_DELTA":
            params = DEFAULT_LOFAR_DELTA
        elif model_type == "FC_DELTA_ON" or model_type == "RNN_DELTA_ON":
            params = DEFAULT_LOFAR_DELTA_ON
        elif model_type == "FC_DELTA_EXPOSURE" or model_type == "RNN_DELTA_EXPOSURE":
            if delta_normalization:
                params = DEFAULT_LOFAR_DELTA_EXPOSURE_DIVNORM
            else:
                params = DEFAULT_LOFAR_DELTA_EXPOSURE
        elif model_type == "FC_FORWARD_STEP" or model_type == "RNN_FORWARD_STEP":
            if delta_normalization:
                params = DEFAULT_LOFAR_FORWARD_DIVNORM
            else:
                params = DEFAULT_LOFAR_FORWARD
        elif model_type == "FC_ANN":
            if delta_normalization:
                params = DEFAULT_LOFAR_ANN_DIVNORM
            else:
                params = DEFAULT_LOFAR_ANN

        elif model_type == "MH_LATENCY":
            params = copy.deepcopy(DEFAULT_HERA_LATENCY_MH)
            params["data_source"]["dataset"] = dataset
            params["encoder"]["exposure"] = 32
            params["model"]["beta"] = 0.1
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "MIXED":
        if model_type == "MH_LATENCY":
            params = copy.deepcopy(DEFAULT_HERA_LATENCY_MH)
            params["data_source"]["dataset"] = dataset
            params["encoder"]["exposure"] = 32
            params["model"]["beta"] = 0.1
        else:
            raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
    params["model"]["type"] = model_type
    params["data_source"]["delta_normalization"] = delta_normalization
    return params
