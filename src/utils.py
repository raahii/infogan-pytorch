import numpy as np
import torch
from PIL import Image


def new_tensor_module():
    return torch.cuda if torch.cuda.is_available() else torch


def current_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


def images_to_numpy(tensor):
    """
    convert pytorch tensor to numpy array

    Parameters
    ----------
    tensor: torch or torch.cuda
        pytorch images tensor
    
    Returns
    ---------
    imgs: numpy.array
        numpy images array
    """

    imgs = tensor.data.cpu().numpy()
    imgs = imgs.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
    imgs = np.clip(imgs, -1, 1)
    imgs = (imgs + 1) / 2 * 255
    imgs = imgs.astype("uint8")

    return imgs


def videos_to_numpy(tensor):
    """
    convert pytorch tensor to numpy array

    Parameters
    ----------
    tensor: torch or torch.cuda
        pytorch tensor in the shape of (batchsize, channel, frames, width, height)
    
    Returns
    ---------
    imgs: numpy.array
        numpy array in the same shape of input tensor
    """
    videos = tensor.data.cpu().numpy()
    videos = np.clip(videos, -1, 1)
    videos = (videos + 1) / 2 * 255
    videos = videos.astype("uint8")

    return videos


def make_video_grid(videos, rows, cols):
    """
    Convert multiple videos to a single rows x cols grid video. 
    It must be len(videos) == rows*cols.

    Parameters
    ----------
    videos: numpy.array
        numpy array in the shape of (batchsize, channel, frames, height, width)

    rows: int
        num rows

    cols: int
        num columns

    Returns
    ----------
    grid_video: numpy.array
        numpy array in the shape of (1, channel, frames, height*rows, width*cols)
    """

    N, C, T, H, W = videos.shape
    assert N == rows * cols

    videos = videos.transpose(1, 2, 0, 3, 4)
    videos = videos.reshape(C, T, rows, cols, H, W)
    videos = videos.transpose(0, 1, 2, 4, 3, 5)
    videos = videos.reshape(C, T, rows * H, cols * W)
    if C == 1:
        videos = np.tile(videos, (3, 1, 1, 1))
    videos = videos[None]

    return videos


def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min())
