import matplotlib.pyplot as plt
import snntorch.spikeplot as spl
import torch


def plot_example_inference(example: torch.tensor, name=str):
    fig, ax = plt.subplots()
    anim = spl.animator(example, fig, ax, interval=100)
    anim.save(f"example_{name}.gif")
    plt.close(fig)
