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
        "beta": 0.7270826938643781,
    },
    "trainer": {
        "epochs": 44,
        "num_nodes": os.getenv("NNODES", 1),
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
        "batch_size": 27,
    },
    "model": {
        "type": "FC_RATE",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "beta": 0.8417118385641611,
    },
    "trainer": {
        "epochs": 34,
        "num_nodes": os.getenv("NNODES", 1),
    },
    "encoder": {
        "method": "RATE",
        "exposure": 16,
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
        "batch_size": 36,
    },
    "model": {
        "type": "FC_DELTA",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 64,
        "beta": 0.7270826938643781,
        "reconstruct_loss": False,
    },
    "trainer": {
        "epochs": 50,
        "num_nodes": os.getenv("NNODES", 1),
        "patience": 100,
    },
    "encoder": {"method": "DELTA", "threshold": 0.1, "off_spikes": True},
}

DEFAULT_LOFAR_DELTA = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_LOFAR_DELTA["data_source"]["dataset"] = "LOFAR"
DEFAULT_TABASCAL_DELTA = copy.deepcopy(DEFAULT_HERA_DELTA)
DEFAULT_TABASCAL_DELTA["data_source"]["dataset"] = "TABASCAL"


def get_default_params(dataset: str, model_type: str):
    if dataset == "HERA":
        if model_type == "FC_LATENCY":
            return DEFAULT_HERA_LATENCY
        elif model_type == "FC_RATE":
            return DEFAULT_HERA_RATE
        elif model_type == "FC_DELTA":
            return DEFAULT_HERA_DELTA
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "LOFAR":
        if model_type == "FC_LATENCY":
            return DEFAULT_LOFAR_LATENCY
        elif model_type == "FC_RATE":
            return DEFAULT_LOFAR_RATE
        elif model_type == "FC_DELTA":
            return DEFAULT_LOFAR_DELTA
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "TABASCAL":
        if model_type == "FC_LATENCY":
            return DEFAULT_TABASCAL_LATENCY
        elif model_type == "FC_RATE":
            return DEFAULT_TABASCAL_RATE
        elif model_type == "FC_DELTA":
            return DEFAULT_TABASCAL_DELTA
        else:
            raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
