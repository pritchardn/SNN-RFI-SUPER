import os
import copy

DEFAULT_HERA_LATENCY = {
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
        "type": "FC_LATENCY",
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
        "method": "LATENCY",
        "exposure": 13,
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
        "exposure": 13,
        "tau": 1.0,
        "normalize": True,
    },
}

DEFAULT_LOFAR_RATE = copy.deepcopy(DEFAULT_HERA_RATE)
DEFAULT_LOFAR_RATE["data_source"]["dataset"] = "LOFAR"
DEFAULT_TABASCAL_RATE = copy.deepcopy(DEFAULT_HERA_RATE)
DEFAULT_TABASCAL_RATE["data_source"]["dataset"] = "TABASCAL"


def get_default_params(dataset: str, model_type: str):
    if dataset == "HERA":
        if model_type == "FC_LATENCY":
            return DEFAULT_HERA_LATENCY
        elif model_type == "FC_RATE":
            return DEFAULT_HERA_RATE
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "LOFAR":
        if model_type == "FC_LATENCY":
            return DEFAULT_LOFAR_LATENCY
        elif model_type == "FC_RATE":
            return DEFAULT_LOFAR_RATE
        else:
            raise ValueError(f"Unknown model type {model_type}")
    elif dataset == "TABASCAL":
        if model_type == "FC_LATENCY":
            return DEFAULT_TABASCAL_LATENCY
        elif model_type == "FC_RATE":
            return DEFAULT_TABASCAL_RATE
        else:
            raise ValueError(f"Unknown model type {model_type}")
    else:
        raise ValueError(f"Unknown dataset {dataset}")
