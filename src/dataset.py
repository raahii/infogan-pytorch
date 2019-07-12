from pathlib import Path
from typing import Union

import torch
import torchvision
import torchvision.transforms as transforms


def new_mnist_dataset(root_path: Union[str, Path]) -> torch.utils.data.Dataset:
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = torchvision.datasets.MNIST(
        root=str(root_path), train=True, download=True, transform=transform
    )

    return dataset


def new_fashion_mnist_dataset(root_path: Union[str, Path]) -> torch.utils.data.Dataset:
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    dataset = torchvision.datasets.FashionMNIST(
        root=str(root_path), train=True, download=True, transform=transform
    )

    return dataset


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid, save_image

    dataset = new_mnist_dataset("data/mnist")

    dataloader = DataLoader(
        dataset,
        batch_size=100,
        num_workers=1,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    # generate a 10x10 grid image
    x, _ = next(iter(dataloader))
    x = make_grid(x, 10, normalize=True)
    torchvision.utils.save_image(x, "dataset.jpg")
