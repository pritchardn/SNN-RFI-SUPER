import os
import numpy as np
import matplotlib.pyplot as plt
import snntorch.spikeplot as spl
import torch


def plot_example_inference(example: torch.tensor, name: str, log_dir: str):
    fig, ax = plt.subplots()
    anim = spl.animator(example, fig, ax, interval=100)
    anim.save(os.path.join(log_dir, f"example_{name}.gif"))
    plt.close(fig)


def plot_example_mask(mask: np.ndarray, name: str, log_dir: str):
    fig, ax = plt.subplots()
    ax.imshow(mask)
    plt.savefig(os.path.join(log_dir, f"example_mask_{name}.png"))
    plt.close(fig)
