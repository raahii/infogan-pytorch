import argparse
import random
import sys
from pathlib import Path
from typing import Dict

import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from dataset import new_mnist_dataset
from logger import Logger
from models import Discriminator, Generator, build_latent_variables
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
    latent_vars = build_latent_variables(config["latent_variables"])
    gen = Generator(latent_vars)
    dis = Discriminator(latent_vars)
    models = {"gen": gen, "dis": dis}

    # prepare optimizers
    opt_gen = create_optimizer(gen.parameters(), **config["optimizer"]["gen"])
    opt_dis = create_optimizer(dis.parameters(), **config["optimizer"]["dis"])
    opts = {"gen": opt_gen, "dis": opt_dis}

    log_path = Path(config["log_path"])
    log_path.mkdir(parents=True, exist_ok=True)
    tb_path = Path(config["tensorboard_path"])
    tb_path.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = Logger(log_path, tb_path)

    # start training
    trainer = Trainer(dataloader, models, opts, config, logger)
    trainer.train()


if __name__ == "__main__":
    main()
