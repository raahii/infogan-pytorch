import argparse
import random
from pathlib import Path
from typing import Dict

import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from dataset import new_mnist_dataset
from models import Discriminator, Generator
from trainer import Trainer


def load_yaml(path: str) -> Dict:
    f = open(path)
    c = yaml.load(f)
    f.close()

    return c


def worker_init_fn(worker_id):
    random.seed(worker_id)


def create_optimizer(params, lr, decay):
    return optim.Adam(params, lr=lr, betas=(0.5, 0.999), weight_decay=decay)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="configs/default.yaml",
        help="training configuration file",
    )
    args = parser.parse_args()
    config = load_yaml(args.config)

    dataset_name = config["dataset"]["name"]
    dataset_path = Path(config["dataset"]["path"]) / dataset_name

    # prepare dataset
    if dataset_name == "mnist":
        dataset = new_mnist_dataset(dataset_path)
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        dataset,
        batch_size=config["dataset"]["batchsize"],
        num_workers=config["dataset"]["n_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # prepare models
    gen, dis = Generator(), Discriminator()
    models = {"gen": gen, "dis": dis}

    # prepare optimizers
    opt_gen = create_optimizer(gen.parameters(), **config["gen"]["optimizer"])
    opt_dis = create_optimizer(dis.parameters(), **config["dis"]["optimizer"])
    opts = {"gen": opt_gen, "dis": opt_dis}

    # start training
    trainer = Trainer(dataloader, models, opts, config)
    trainer.train()


if __name__ == "__main__":
    main()
