import os

DEFAULT_CONFIG = {
    "data_source": {
        "data_path": "./data",
        "limit": 0.1,
        "patch_size": 32,
        "stride": 32,
        "dataset": "HERA",
    },
    "dataset": {
        "batch_size": 18,
    },
    "model": {
        "type": "FC_LATENCY",
        "num_inputs": 32,
        "num_hidden": 128,
        "num_outputs": 32,
        "beta": 0.8976593051095902,
    },
    "trainer": {
        "epochs": 60,
        "num_nodes": os.getenv("NNODES", 1),
    },
    "encoder": {
        "method": "LATENCY",
        "exposure": 25,
        "tau": 1.0,
        "normalize": True,
    },
}
