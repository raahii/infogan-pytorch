import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

import loss
from dataset import new_mnist_dataset
from logger import Logger
from models import (DHead, Discriminator, Generator, QHead,
                    build_latent_variables)
from trainer import Trainer


def load_yaml(path: str) -> Dict[str, Any]:
    f = open(path)
    c = yaml.load(f, Loader=yaml.FullLoader)
    f.close()

    return c


def worker_init_fn(worker_id: int):
    random.seed(worker_id)


def create_optimizer(models: List[nn.Module], lr: float, decay: float):
    params: List[torch.Tensor] = []
    for m in models:
        params += list(m.parameters())
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
    configs = load_yaml(args.config)

    dataset_name = configs["dataset"]["name"]
    dataset_path = Path(configs["dataset"]["path"]) / dataset_name

    # prepare dataset
    if dataset_name == "mnist":
        dataset = new_mnist_dataset(dataset_path)
    else:
        raise NotImplementedError

    dataloader = DataLoader(
        dataset,
        batch_size=configs["dataset"]["batchsize"],
        num_workers=configs["dataset"]["n_workers"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    # prepare models
    latent_vars = build_latent_variables(configs["latent_variables"])
    gen, dis = (
        Generator(latent_vars),
        Discriminator(latent_vars, configs["models"]["dis"]),
    )
    dhead, qhead = DHead(), QHead(latent_vars)
    models = {"gen": gen, "dis": dis, "dhead": dhead, "qhead": qhead}

    # prepare optimizers
    opt_gen = create_optimizer([gen, qhead], **configs["optimizer"]["gen"])
    opt_dis = create_optimizer([dis, dhead], **configs["optimizer"]["dis"])
    opts = {"gen": opt_gen, "dis": opt_dis}

    # prepare directories
    log_path = Path(configs["log_path"])
    log_path.mkdir(parents=True, exist_ok=True)
    tb_path = Path(configs["tensorboard_path"])
    tb_path.mkdir(parents=True, exist_ok=True)

    # initialize logger
    logger = Logger(log_path, tb_path)

    # initialize losses
    losses = {"adv": loss.AdversarialLoss(), "info": loss.InfoGANLoss(latent_vars)}

    # start training
    trainer = Trainer(dataloader, models, opts, losses, configs["training"], logger)
    trainer.train()


if __name__ == "__main__":
    main()
