from pathlib import Path
from typing import Union

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def new_mnist_dataset(root_path: Union[str, Path]) -> torch.utils.data.Dataset:
    transform = transforms.Compose(
        [transforms.Resize(28), transforms.CenterCrop(28), transforms.ToTensor()]
    )

    dataset = torchvision.datasets.MNIST(
        root=str(root_path), train=True, download=True, transform=transform
    )

    return dataset


if __name__ == "__main__":
    import pdb

    pdb.set_trace()
    dataset = new_mnist_dataset("data/mnist")

    dataloader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )
