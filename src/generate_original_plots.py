import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiment import data_source_from_config
from generate_plots import setup_config


def load_dataset_examples(config: dict, limit: int) -> [np.ndarray, np.ndarray]:
    data_source = data_source_from_config(config["data_source"])
    data_source.load_data()
    test_x = data_source.fetch_test_x()
    test_y = data_source.fetch_test_y()

    test_x = test_x[:limit]
    test_y = test_y[:limit]
    return test_x, test_y


def plot_original_example(
        test_x, test_y, outdir, i, title
):
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
    image1 = ax1.imshow(np.moveaxis(test_x, 0, -1))
    image2 = ax2.imshow(np.moveaxis(test_y, 0, -1))
    plt.colorbar(image1, location="right", shrink=0.9)
    plt.colorbar(image2, location="right", shrink=0.9, ticks=[0, 1])
    fig.text(0.5, 0.1, "Time [s]", ha="center", va="center", fontsize=16)
    ax.set_ylabel("Frequency bin")
    for ax in [ax1, ax2]:
        ax.invert_yaxis()
    # plt.subplots_adjust(bottom=0.2, top=0.8, left=0.1, right=0.9)
    plt.subplots_adjust(bottom=0.2, top=0.8, wspace=0.3)
    plt.savefig(f"{outdir}{os.sep}original_{title}_example_{i}.png", bbox_inches="tight")
    plt.close()


def main(outdir: str):
    # Get config
    config, frequency_width, used_exposure = setup_config(
        "FC_LATENCY", 4, None, 1
    )
    config["data_source"]["patch_size"] = None
    # Open Dataset
    test_x, test_y = load_dataset_examples(config, -1)
    # For each spectrogram
    for i in tqdm(range(len(test_x))):
        plot_original_example(
            test_x[i],
            test_y[i],
            outdir,
            i,
            f"LOFAR_DIVNORM_TEST"
        )


if __name__ == "__main__":
    main("./plots_original")
