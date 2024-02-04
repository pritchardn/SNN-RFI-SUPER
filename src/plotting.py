import matplotlib.pyplot as plt
import snntorch.spikeplot as spl
import torch
import os


def plot_example_inference(example: torch.tensor, name: str, log_dir: str):
    fig, ax = plt.subplots()
    anim = spl.animator(example, fig, ax, interval=100)
    anim.save(os.path.join(log_dir, f"example_{name}.gif"))
    plt.close(fig)
