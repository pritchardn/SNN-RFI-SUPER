"""
This script generates plots for the HERA dataset.
"""
import multiprocessing
import os

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import get_default_params
from experiment import data_source_from_config, encoder_from_config


def load_dataset_examples(config: dict, limit: int):
    data_source = data_source_from_config(config["data_source"])
    data_source.load_data()
    # Get N examples from the dataset
    test_x = data_source.fetch_test_x()
    test_y = data_source.fetch_test_y()
    # Get subset
    test_x = test_x[:limit]
    test_y = test_y[:limit]
    return test_x, test_y


def plot_example_original(x, y, i, title: str, outdir="./"):
    fig = plt.figure(figsize=(10, 5))
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc("axes", labelsize=16)
    plt.rc("xtick", labelsize=16)
    plt.rc("ytick", labelsize=16)
    plt.tight_layout()
    ax = fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    image1 = ax1.imshow(np.moveaxis(x, 0, -1))
    image2 = ax2.imshow(np.moveaxis(y, 0, -1))
    plt.colorbar(image1, location="right", shrink=0.9)
    plt.colorbar(image2, location="right", shrink=0.9, ticks=[0, 1])
    fig.text(0.5, 0.1, "Time [s]", ha="center", va="center", fontsize=16)
    ax.set_ylabel("Frequency bin")
    for ax in [ax1, ax2]:
        ax.invert_yaxis()
    # plt.subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.9)
    plt.subplots_adjust(bottom=0.2, top=0.8, wspace=0.3)
    plt.savefig(os.path.join(outdir, f"original_{title}_example_{i}.png"), bbox_inches="tight")
    plt.close()


def plot_example_raster(
        spike_x, frequency_width, stride, exposure, i, title: str, mode=1, outdir="./"
):
    # plt.tight_layout()
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc("axes", labelsize=10 * mode)
    plt.rc("xtick", labelsize=8 * mode)
    plt.rc("ytick", labelsize=8 * mode)
    plt.figure(figsize=(10, 5))
    example = spike_x
    example = example.squeeze(1)  # Remove channel dimension
    out = np.zeros((frequency_width, stride * exposure))
    for t in range(example.shape[-1]):  # t
        out[:, t * exposure: (t + 1) * exposure] = np.moveaxis(example[:, :, t], 0, -1)
    if min(spike_x.flatten()) < 0:
        ticks = [-1, 0, 1]
        cmap = plt.get_cmap("viridis", 3)
        # cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist = [cmap(1), cmap(0), cmap(2)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, cmap.N
        )
    else:
        ticks = [0, 1]
        cmap = plt.get_cmap("viridis", 3)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # cmaplist[0] = (1.0, 1.0, 1.0, 1.0)
        cmaplist = [cmap(0), cmap(2)]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "Custom cmap", cmaplist, cmap.N - 1
        )
    plt.imshow(out, cmap=cmap)
    plt.gca().invert_yaxis()
    plt.ylabel("Frequency bin")
    plt.xlabel("Time [s]")

    plt.colorbar(location="right", ticks=ticks, shrink=0.5 * mode)
    plt.savefig(os.path.join(outdir, f"raster_{title}_example_{i}.png"), bbox_inches="tight")
    plt.close()


def plot_example(x, y, spike_x, frequency_width, stride, exposure, i, title: str, outdir="./"):
    fig = plt.figure()
    plt.tight_layout()
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    example = spike_x
    example = example.squeeze(1)  # Remove channel dimension
    out = np.zeros((frequency_width, stride * exposure))
    for t in range(example.shape[-1]):  # t
        out[:, t * exposure: (t + 1) * exposure] = np.moveaxis(example[:, :, t], 0, -1)
    ax1.imshow(out)
    ax1.set_title("Spike Train")
    ax1.set_xlabel("Time [s]")
    pic = ax2.imshow(np.moveaxis(x, 0, -1))
    fig.colorbar(pic, ax=ax2, location="right")
    ax2.set_title("Original Spectrogram")
    ax3.imshow(np.moveaxis(y, 0, -1))
    ax3.set_title("RFI Mask")
    plt.title(f"Example {i}")
    for ax in [ax1, ax2, ax3]:
        ax.invert_yaxis()
    plt.savefig(os.path.join(outdir, f"plot_{title}_example_{i}.png"), bbox_inches="tight")
    plt.close()


def setup_config(model, exposure, exposure_mode, stride):
    config = get_default_params("HERA", model, 128, exposure_mode)
    frequency_width = stride
    if model == "FC_FORWARD_STEP":
        frequency_width *= 2
    if model == "FC_DELTA":
        exposure = 1
    config["encoder"]["exposure"] = exposure
    return config, frequency_width, exposure


def main_single(model, exposure_mode, stride, exposure, limit: int = 10):
    # Load HERA data
    config, frequency_width, exposure = setup_config(
        model, exposure, exposure_mode, stride
    )
    test_x, test_y = load_dataset_examples(config, limit)
    # Create converter
    encoder = encoder_from_config(config["encoder"])
    spike_x = encoder.encode_x(test_x)
    # Plot examples
    for i in range(limit):
        plot_example(
            test_x[i],
            test_y[i],
            spike_x[i],
            frequency_width,
            stride,
            exposure,
            i,
            f"{model}" + f"_{exposure_mode}" if exposure_mode else "",
        )


def main_mini(plot_mode, model, test_x, test_y, spike_x, frequency_width, stride, used_exposure, exposure_mode, i,
              outdir):
    title = f"{model}" + (f"_{exposure_mode}" if exposure_mode else "")
    plot_example(
        test_x,
        test_y,
        spike_x,
        frequency_width,
        stride,
        used_exposure,
        i,
        title,
        outdir=outdir
    )
    plot_example_raster(
        spike_x,
        frequency_width,
        stride,
        used_exposure,
        i,
        title,
        mode=plot_mode,
        outdir=outdir
    )
    plot_example_original(test_x, test_y, i, title, outdir=outdir)


def main_all(stride, exposure, limit: int = 10, outdir="./"):
    test_x, test_y = None, None
    with multiprocessing.Pool() as pool:
        for model, exposure_mode, plot_mode in tqdm([
            ("FC_LATENCY", None, 1),
            ("FC_RATE", None, 1),
            ("FC_DELTA", None, 2),
            ("FC_DELTA_EXPOSURE", None, 1),
            ("FC_FORWARD_STEP", "first", 2),
            ("FC_FORWARD_STEP", "direct", 2),
            ("FC_FORWARD_STEP", "latency", 2),
        ]):
            print(model)
            config, frequency_width, used_exposure = setup_config(
                model, exposure, exposure_mode, stride
            )
            if test_x is None or test_y is None:
                test_x, test_y = load_dataset_examples(config, limit)
            encoder = encoder_from_config(config["encoder"])
            spike_x = encoder.encode_x(test_x)
            args = []
            for i in range(limit):
                args.append(
                    (
                    plot_mode, model, test_x[i], test_y[i], spike_x[i], frequency_width, stride, used_exposure, exposure_mode,
                    i, outdir))

            pool.starmap(main_mini, args)


if __name__ == "__main__":
    model = "FC_FORWARD_STEP"
    exposure_mode = "first"
    stride = 32
    exposure = 4
    # main_single(model, exposure_mode, stride, exposure, limit=10)
    main_all(stride, exposure, limit=1280, outdir="./example_plots")
