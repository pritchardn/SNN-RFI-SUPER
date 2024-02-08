import os
from config import DEFAULT_CONFIG
from experiment import Experiment


def main():
    config = DEFAULT_CONFIG
    config["data_source"]["data_path"] = os.getenv("DATA_PATH", config["data_source"]["data_path"])
    root_dir = os.getenv("OUTPUT_DIR", "./")
    experiment = Experiment(root_dir=root_dir)
    experiment.from_config(config)
    experiment.prepare()
    experiment.train()
    experiment.evaluate()


if __name__ == "__main__":
    main()
