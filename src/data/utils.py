import numpy as np
import torch


def extract_patches(data: torch.Tensor, kernel_size: int, stride: int):
    """
    Extracts patches from a tensor. Implements the same functionality as found in tensorflow.
    """
    _, channels, _, _ = data.shape
    # Extract patches
    patches = data.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(
        -1, channels, kernel_size, kernel_size
    )
    return patches


def reconstruct_patches(images: np.array, original_size: int, kernel_size: int):
    """
    Reconstructs patches into images. Implements the same functionality as found in tensorflow.
    Transposes the images to match the tensorflow implementation but returns the images in the
    original format.
    """
    transposed = images.transpose(0, 3, 2, 1)
    n_patches = original_size // kernel_size
    recon = np.empty(
        [
            images.shape[0] // n_patches ** 2,
            kernel_size * n_patches,
            kernel_size * n_patches,
            images.shape[1],
        ]
    )

    start, counter, indx, batch = 0, 0, 0, []

    for i in range(n_patches, images.shape[0] + 1, n_patches):
        batch.append(
            np.reshape(
                np.stack(transposed[start:i, ...], axis=0),
                (n_patches * kernel_size, kernel_size, images.shape[1]),
            )
        )
        start = i
        counter += 1
        if counter == n_patches:
            recon[indx, ...] = np.hstack(batch)
            indx += 1
            counter, batch = 0, []

    return recon.transpose(0, 3, 2, 1)


def filter_noiseless_patches(x_data: np.ndarray, y_data: np.ndarray) -> (np.ndarray, np.ndarray):
    index_vales = np.any(y_data, axis=(1, 2, 3))
    out_x = x_data[index_vales]
    out_y = y_data[index_vales]
    return out_x, out_y
