from config import get_default_params
from experiment import data_source_from_config, encoder_from_config
import numpy as np
import matplotlib.pyplot as plt


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


def plot_example_original(x, y, i, title: str):
    fig = plt.figure()
    plt.tight_layout()
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax1.imshow(np.moveaxis(x, 0, -1))
    ax2.imshow(np.moveaxis(y, 0, -1))
    ax2.set_title("RFI Mask")
    plt.title(f"Example {i}")
    plt.savefig(f"original_{title}_example_{i}.png", bbox_inches='tight')
    plt.close()


def plot_example_raster(spike_x, frequency_width, stride, exposure, i, title: str):
    plt.tight_layout()
    example = spike_x
    example = example.squeeze(1)  # Remove channel dimension
    out = np.zeros((frequency_width, stride * exposure))
    for t in range(example.shape[-1]):  # t
        out[:, t * exposure: (t + 1) * exposure] = np.moveaxis(example[:, :, t], 0, -1)
    plt.imshow(out)
    plt.savefig(f"raster_{title}_example_{i}.png", bbox_inches='tight')
    plt.close()


def plot_example(x, y, spike_x, frequency_width, stride, exposure, i, title: str):
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
    ax2.imshow(np.moveaxis(x, 0, -1))
    ax2.set_title("Original Spectrogram")
    ax3.imshow(np.moveaxis(y, 0, -1))
    ax3.set_title("RFI Mask")
    plt.title(f"Example {i}")
    plt.savefig(f"plot_{title}_example_{i}.png", bbox_inches='tight')
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
    config, frequency_width, exposure = setup_config(model, exposure, exposure_mode, stride)
    test_x, test_y = load_dataset_examples(config, limit)
    # Create converter
    encoder = encoder_from_config(config["encoder"])
    spike_x = encoder.encode_x(test_x)
    # Plot examples
    for i in range(limit):
        plot_example(test_x[i], test_y[i], spike_x[i], frequency_width, stride, exposure, i,
                     f"{model}" + f"_{exposure_mode}" if exposure_mode else "")


def main_all(stride, exposure, limit: int = 10):
    test_x, test_y = None, None
    for model, exposure_mode in [("FC_LATENCY", None), ("FC_RATE", None), ("FC_DELTA", None),
                                 ("FC_FORWARD_STEP", "first"), ("FC_FORWARD_STEP", "direct"),
                                 ("FC_FORWARD_STEP", "latency")]:
        print(model)
        config, frequency_width, used_exposure = setup_config(model, exposure, exposure_mode,
                                                              stride)
        if test_x is None or test_y is None:
            test_x, test_y = load_dataset_examples(config, limit)
        encoder = encoder_from_config(config["encoder"])
        spike_x = encoder.encode_x(test_x)
        for i in range(limit):
            title = f"{model}" + (f"_{exposure_mode}" if exposure_mode else "")
            plot_example(test_x[i], test_y[i], spike_x[i], frequency_width, stride, used_exposure,
                         i, title)
            plot_example_raster(spike_x[i], frequency_width, stride, used_exposure, i, title)
            plot_example_original(test_x[i], test_y[i], i, title)


if __name__ == "__main__":
    model = "FC_FORWARD_STEP"
    exposure_mode = "first"
    stride = 32
    exposure = 4
    # main_single(model, exposure_mode, stride, exposure, limit=10)
    main_all(stride, exposure, limit=10)
