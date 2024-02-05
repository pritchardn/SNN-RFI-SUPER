from config import DEFAULT_CONFIG
from experiment import Experiment


def main():
    config = DEFAULT_CONFIG
    experiment = Experiment()
    experiment.from_config(config)
    experiment.prepare()
    experiment.train()
    experiment.evaluate()


if __name__ == "__main__":
    main()
