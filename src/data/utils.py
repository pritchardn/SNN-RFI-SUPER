"""
Utility functions for data processing.
"""

from typing import Union

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
    # Check for DoP entry
    if images.shape[-1] != images.shape[-2]:
        images = images[:, :, :-1, :]
    transposed = images.transpose(0, 3, 2, 1)
    n_patches = original_size // kernel_size
    recon = np.empty(
        [
            images.shape[0] // n_patches**2,
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


def filter_noiseless_patches(
    x_data: np.ndarray, y_data: np.ndarray
) -> (np.ndarray, np.ndarray):
    index_vales = np.any(y_data, axis=(1, 2, 3))
    out_x = x_data[index_vales]
    out_y = y_data[index_vales]
    return out_x, out_y


def ensure_tflow(data: np.ndarray):
    if data.shape[1] == 1:
        return np.moveaxis(data, 1, -1)
    return data


def _decode_delta_inference_numpy(spike_hat: np.ndarray) -> np.ndarray:
    # Assuming [exp, N, C, freq * 2, time]
    inference = np.squeeze(spike_hat, axis=0)  # [N, C, freq, time]
    out = np.zeros(
        (
            inference.shape[0],
            inference.shape[1],
            inference.shape[2] // 2,
            inference.shape[3],
        ),
        dtype=inference.dtype,
    )
    # Copy even frequency channels
    out[:, :, :, :] = inference[:, :, ::2, :]
    # Copy odd frequency channels but spikes are converted to off spikes
    out[:, :, :, :] = np.where(inference[:, :, 1::2, :] == 1, -1, out[:, :, :, :])
    for i in range(out.shape[0]):
        for j in range(out.shape[2]):
            flag = False
            for k in range(out.shape[3]):
                curr = out[i, :, j, k]
                if flag:
                    if curr == 0 or curr == 1:
                        out[i, :, j, k] = 1
                    elif curr == -1:
                        flag = False
                        out[i, :, j, k] = 0
                else:
                    if curr == 1:
                        flag = True
                        out[i, :, j, k] = 1
                    elif curr == -1:
                        out[i, :, j, k] = 0

    return out  # [N, C, freq, time]


def _decode_delta_inference_torch(spike_hat: torch.Tensor) -> torch.Tensor:
    # Assuming [exp, N, C, freq * 2, time]
    inference = torch.squeeze(spike_hat, dim=0)  # [N, C, freq, time]
    out = torch.zeros(
        (
            inference.shape[0],
            inference.shape[1],
            inference.shape[2] // 2,
            inference.shape[3],
        ),
        dtype=inference.dtype,
        device=inference.device,
    )
    # Copy even frequency channels
    out[:, :, :, :] = inference[:, :, ::2, :]
    # Copy odd frequency channels but spikes are converted to off spikes
    out[:, :, :, :] = torch.where(inference[:, :, 1::2, :] == 1, -1, out[:, :, :, :])

    for i in range(out.shape[0]):
        for j in range(out.shape[2]):
            flag = False
            for k in range(out.shape[3]):
                curr = out[i, :, j, k]
                if flag:
                    if curr == 0 or curr == 1:
                        out[i, :, j, k] = 1
                    elif curr == -1:
                        flag = False
                        out[i, :, j, k] = 0
                else:
                    if curr == 1:
                        flag = True
                        out[i, :, j, k] = 1
                    elif curr == -1:
                        out[i, :, j, k] = 0

    return out


def decode_delta_inference(
    spike_hat, use_numpy: bool
) -> Union[np.ndarray, torch.Tensor]:
    if use_numpy:
        return _decode_delta_inference_numpy(spike_hat)
    return _decode_delta_inference_torch(spike_hat)


def test_train_split(data, masks, train_size: float = 0.8):
    # Split the training data into training and test sets
    train_size = int(train_size * data.shape[0])
    indices = np.random.permutation(data.shape[0])
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_x, test_x = data[train_indices], data[test_indices]
    train_y, test_y = masks[train_indices], masks[test_indices]
    return train_x, train_y, test_x, test_y


def extract_polarization(data, polarization: int):
    return np.expand_dims(data[:, polarization, :, :], 1)


def expand_polarization(data):
    # [N, C, freq, time] -> [N, C * freq, time]
    return np.expand_dims(np.reshape(data, (data.shape[0], -1, data.shape[-1])), 1)
