"""
This script evaluates a trained model using a checkpoint.
"""

import json
import os

from experiment import Experiment


def main():
    root_dir = os.getenv("OUTPUT_DIR", "./")
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", None)
    experiment = Experiment(root_dir=root_dir)
    experiment.from_checkpoint(checkpoint_dir)
    experiment.prepare()
    print("Preparation complete")
    metrics = experiment.evaluate(plot=True)
    print("Evaluation complete")
    print(experiment.trainer.log_dir)
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()
