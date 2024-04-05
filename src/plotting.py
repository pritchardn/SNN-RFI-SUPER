"""
This module contains functions for plotting the results of the inference process.
"""
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


def plot_final_examples(
    y_true: np.ndarray, y_pred: np.ndarray, name: str, log_dir: str
):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(y_true)
    axs[0].set_title("Mask")
    axs[1].imshow(y_pred)
    axs[1].set_title("Decoded Output")
    plt.savefig(os.path.join(log_dir, f"final_{name}.png"))
    plt.close(fig)
