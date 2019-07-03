from typing import Any, Dict

import torch
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    f = open(path)
    c = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    return c


def current_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")
