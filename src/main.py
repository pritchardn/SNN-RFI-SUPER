import os

from config import get_default_params
from experiment import Experiment


def main():
    model_type = os.getenv("MODEL_TYPE", "FC_LATENCY")
    dataset = os.getenv("DATASET", "HERA")
    num_hidden = int(os.getenv("NUM_HIDDEN", 128))
    num_layers = int(os.getenv("NUM_LAYERS", 2))
    exposure_mode = os.getenv("EXPOSURE_MODE", None)
    plot = bool(os.getenv("PLOT", False))
    config = get_default_params(dataset, model_type, num_hidden, exposure_mode)
    config["data_source"]["data_path"] = os.getenv(
        "DATA_PATH", config["data_source"]["data_path"]
    )
    config["model"]["num_layers"] = num_layers
    config["data_source"]["limit"] = float(
        os.getenv("LIMIT", config["data_source"]["limit"])
    )
    config["encoder"]["exposure"] = int(
        os.getenv("EXPOSURE", config["encoder"].get("exposure", 1))
    )
    root_dir = os.getenv("OUTPUT_DIR", "./")
    experiment = Experiment(root_dir=root_dir)
    experiment.from_config(config)
    experiment.prepare()
    print("Preparation complete")
    experiment.train()
    print("Training complete")
    experiment.evaluate(plot)
    print("Evaluation complete")


if __name__ == "__main__":
    main()
