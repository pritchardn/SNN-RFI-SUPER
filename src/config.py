DEFAULT_CONFIG = {
    "data_source": {
        "data_path": "./data",
        "limit": 0.1,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 32,
    },
    "model": {
        "type": "FC_LATENCY",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "beta": 0.95,
    },
    "trainer": {
        "epochs": 50,
    },
    "encoder": {
        "method": "LATENCY",
        "exposure": 8,
        "tau": 1.0,
        "normalize": True,
    },
}
