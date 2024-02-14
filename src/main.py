import os

from config import get_default_params
from experiment import Experiment


def main():
    config = get_default_params("TABASCAL", "FC_LATENCY")
    config["data_source"]["data_path"] = os.getenv(
        "DATA_PATH", config["data_source"]["data_path"]
    )
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
