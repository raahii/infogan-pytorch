import numpy as np
import torch
from PIL import Image


def current_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")
