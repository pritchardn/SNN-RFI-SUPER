import os
import copy

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
        "num_layers": 2,
        "beta": 0.7270826938643781,
    },
    "trainer": {
        "epochs": 44,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "LATENCY",
        "exposure": 6,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_LOFAR_LATENCY = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_LOFAR_LATENCY["data_source"]["dataset"] = "LOFAR"
DEFAULT_LOFAR_LATENCY["dataset"]["batch_size"] = 47
DEFAULT_LOFAR_LATENCY["model"]["beta"] = 0.5579856182276725
DEFAULT_LOFAR_LATENCY["trainer"]["epochs"] = 32
DEFAULT_LOFAR_LATENCY["encoder"]["exposure"] = 48
DEFAULT_TABASCAL_LATENCY = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_TABASCAL_LATENCY["data_source"]["dataset"] = "TABASCAL"

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

DEFAULT_LOFAR_RATE = copy.deepcopy(DEFAULT_HERA_RATE)
DEFAULT_LOFAR_RATE["data_source"]["dataset"] = "LOFAR"
DEFAULT_TABASCAL_RATE = copy.deepcopy(DEFAULT_HERA_RATE)
DEFAULT_TABASCAL_RATE["data_source"]["dataset"] = "TABASCAL"

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

DEFAULT_LOFAR_DELTA = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_LOFAR_DELTA["data_source"]["dataset"] = "LOFAR"
DEFAULT_TABASCAL_DELTA = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_TABASCAL_DELTA["data_source"]["dataset"] = "TABASCAL"

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

DEFAULT_LOFAR_DELTA_ON = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_LOFAR_DELTA_ON["data_source"]["dataset"] = "LOFAR"
DEFAULT_TABASCAL_DELTA_ON = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_TABASCAL_DELTA_ON["data_source"]["dataset"] = "TABASCAL"

DEFAULT_HERA_FORWARD = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 79,
    },
    "model": {
        "type": "FC_FORWARD_STEP",
        "num_inputs": 64,
        "num_hidden": 128,
        "num_outputs": 32,
        "num_layers": 2,
        "beta": 0.948840453252918,
    },
    "trainer": {
        "epochs": 100,
        "num_nodes": int(os.getenv("NNODES", 1)),
    },
    "encoder": {
        "method": "FORWARDSTEP",
        "exposure": 41,
        "tau": 1.0,
        "threshold": 0.1,
        "normalize": True,
        "exposure_mode": "first",
    },
}

DEFAULT_LOFAR_FORWARD = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_LOFAR_FORWARD["data_source"]["dataset"] = "LOFAR"
DEFAULT_TABASCAL_FORWARD = copy.deepcopy(DEFAULT_HERA_LATENCY)
DEFAULT_TABASCAL_FORWARD["data_source"]["dataset"] = "TABASCAL"

DEFAULT_HERA_ANN = {
    "data_source": {
        "data_path": "./data",
        "limit": 1.0,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 80,
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


def get_default_params(dataset: str, model_type: str, model_size: int = 128):
    if dataset == "HERA":
        if model_type == "FC_LATENCY":
            if model_size == 128:
                return DEFAULT_HERA_LATENCY
            elif model_size == 256:
                params = copy.deepcopy(DEFAULT_HERA_LATENCY)
                params["dataset"]["batch_size"] = 74
                params["model"]["num_hidden"] = 256
                params["model"]["beta"] = 0.8333011064675617
                params["trainer"]["epochs"] = 83
                params["encoder"]["exposure"] = 20
                return params
            raise ValueError(f"Unknown model size {model_size}")
        elif model_type == "FC_RATE":
            return DEFAULT_HERA_RATE
        elif model_type == "FC_DELTA":
            return DEFAULT_HERA_DELTA
        elif model_type == "FC_DELTA_ON":
            return DEFAULT_HERA_DELTA_ON
        elif model_type == "FC_FORWARD_STEP":
            return DEFAULT_HERA_FORWARD
        elif model_type == "FC_ANN":
            return DEFAULT_HERA_ANN
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "LOFAR":
        if model_type == "FC_LATENCY":
            return DEFAULT_LOFAR_LATENCY
        elif model_type == "FC_RATE":
            return DEFAULT_LOFAR_RATE
        elif model_type == "FC_DELTA":
            return DEFAULT_LOFAR_DELTA
        elif model_type == "FC_DELTA_ON":
            return DEFAULT_LOFAR_DELTA_ON
        elif model_type == "FC_FORWARD_STEP":
            return DEFAULT_LOFAR_FORWARD
        elif model_type == "FC_ANN":
            return DEFAULT_HERA_ANN
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "TABASCAL":
        if model_type == "FC_LATENCY":
            return DEFAULT_TABASCAL_LATENCY
        elif model_type == "FC_RATE":
            return DEFAULT_TABASCAL_RATE
        elif model_type == "FC_DELTA":
            return DEFAULT_TABASCAL_DELTA
        elif model_type == "FC_DELTA_ON":
            return DEFAULT_TABASCAL_DELTA_ON
        elif model_type == "FC_FORWARD_STEP":
            return DEFAULT_TABASCAL_FORWARD
        elif model_type == "FC_ANN":
            return DEFAULT_HERA_ANN
        else:
            raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
