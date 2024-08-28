import json
import os

import matplotlib.pyplot as plt
import numpy as np
import snntorch.spikeplot as splt
import torch
from tqdm import tqdm

cmap_name = "viridis"
interval = 250


def animate_patch(patch, i, output_dir):
    fig, ax = plt.subplots()
    anim = splt.animator(
        torch.from_numpy(np.moveaxis(np.copy(patch), 1, -1)),
        fig,
        ax,
        interval=interval,
        cmap=cmap_name,
    )
    anim.save(os.path.join(output_dir, f"patch_{i}.gif"))


def animate_printer(spectrogram, i, num_patches, output_dir, encoding_mode):
    exposure = spectrogram.shape[0]
    # create numpy array with shape (exposure * num_patches, **exposure.shape)
    out = np.zeros((exposure * num_patches, *spectrogram.shape[1:]))
    patch_size = spectrogram.shape[-1] // num_patches
    decoded_inference = spectrogram[:-1, :, :, :].sum(axis=0)
    if encoding_mode == "LATENCY":
        decoded_inference[decoded_inference > 1] = 1
    for j in range(num_patches):
        mini_frame = np.zeros(spectrogram.shape)
        mini_frame[:, :, :, j * patch_size : (j + 1) * patch_size] = spectrogram[
            :, :, :, j * patch_size : (j + 1) * patch_size
        ]
        mini_frame[:, :, :, : j * patch_size] = decoded_inference[
            :, :, : j * patch_size
        ]
        out[j * exposure : (j + 1) * exposure, ...] = mini_frame
    fig, ax = plt.subplots()
    animate_data = np.moveaxis(out, 1, -1)
    anim = splt.animator(
        torch.from_numpy(np.moveaxis(out, 1, -1)),
        fig,
        ax,
        interval=interval,
        cmap=cmap_name,
    )
    anim.save(os.path.join(output_dir, f"sequence_{i}.gif"))


def reconstruct_patches_time(
    data: np.array, orig_size: int, patch_size: int, encoding_mode: str
):
    """
    Works like reconstruct_patches but includes the time-axis present in spiking data.
    Input: [t, N, C, F, T]
    Output: [t, N, C, F, T]
    :param data:
    :param orig_size:
    :param patch_size:
    :return:
    """
    transposed = data.transpose(1, 4, 3, 2, 0)
    n_patches = orig_size // patch_size
    recon = np.empty(
        [
            data.shape[1] // n_patches**2,
            patch_size * n_patches,
            patch_size * n_patches * (2 if encoding_mode == "DELTA" else 1),
            data.shape[2],
            data.shape[0],
        ]
    )
    start, counter, indx, batch = 0, 0, 0, []
    for i in range(n_patches, data.shape[1] + 1, n_patches):
        batch.append(
            np.reshape(
                np.stack(transposed[start:i, ...], axis=0),
                (
                    n_patches * patch_size,
                    patch_size * (2 if encoding_mode == "DELTA" else 1),
                    data.shape[2],
                    data.shape[0],
                ),
            )
        )
        start = i
        counter += 1
        if counter == n_patches:
            recon[indx, ...] = np.hstack(batch)
            indx += 1
            counter, batch = 0, []
    return recon.transpose(4, 0, 3, 2, 1)


def main(input_dir: str):
    file_dir = os.path.join(input_dir, "full_spike_hat.npy")
    config_file = os.path.join(input_dir, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    encoding_mode = config["encoder"]["method"]
    data = np.load(file_dir)
    original_size = 512
    patch_size = 32
    num_patches = original_size // patch_size
    reconstructed_patches = reconstruct_patches_time(
        data, original_size, patch_size, encoding_mode
    )
    assert reconstructed_patches.shape == (
        data.shape[0],
        140,
        1,
        512 * (2 if encoding_mode == "DELTA" else 1),
        512,
    )
    if encoding_mode == "DELTA":
        reconstructed_patches = np.flip(reconstructed_patches, axis=3)

    for i in tqdm(range(max(10, reconstructed_patches.shape[1]))):
        animate_patch(reconstructed_patches[:, i, ...], i, input_dir)
        animate_printer(
            reconstructed_patches[:, i, ...], i, num_patches, input_dir, encoding_mode
        )


if __name__ == "__main__":
    for filename in [  # "lightning_logs_icons_plotting/version_0/",
        # "lightning_logs_icons_plotting/version_1/",
        "lightning_logs_icons_plotting/version_2/"
    ]:
        main(filename)
