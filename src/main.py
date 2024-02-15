import os

from config import get_default_params
from experiment import Experiment


def main():
    model_type = os.getenv("MODEL_TYPE", "FC_LATENCY")
    dataset = os.getenv("DATASET", "HERA")
    config = get_default_params(model_type, dataset)
    config["data_source"]["data_path"] = os.getenv(
        "DATA_PATH", config["data_source"]["data_path"]
    )
    config["data_source"]["limit"] = float(os.getenv("LIMIT", config["data_source"]["limit"]))
    root_dir = os.getenv("OUTPUT_DIR", "./")
    experiment = Experiment(root_dir=root_dir)
    experiment.from_config(config)
    experiment.prepare()
    print("Preparation complete")
    experiment.train()
    print("Training complete")
    experiment.evaluate()
    print("Evaluation complete")


if __name__ == "__main__":
    main()
