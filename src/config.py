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
