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


def plot_image_patch(
    image: np.ndarray, filename_prefix: str, output_dir: str, cbar=False
):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    plt.ylabel("Frequency Bins")
    plt.xlabel("Time [s]")
    if cbar:
        plt.colorbar(location="right", shrink=0.8)
    plt.gca().invert_yaxis()
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_image.png"),
        bbox_inches="tight", dpi=300
    )
    plt.close("all")


def plot_final_examples(
    x_orig: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, name: str, log_dir: str
):
    # plot original
    plot_image_patch(x_orig, f"{name}_image", log_dir, cbar=True)
    # plot decoded mask
    plot_image_patch(y_pred, f"{name}_inference", log_dir)
    # plot real mask
    plot_image_patch(y_true, f"{name}_mask", log_dir)
