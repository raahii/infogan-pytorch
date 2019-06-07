import random
from typing import Dict

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from logger import Logger


class Trainer(object):
    def __init__(
        self,
        dataloader: DataLoader,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Optimizer],
        configs: Dict,
    ):

        self.dataloader = dataloader
        self.models = models
        self.optimizers = optimizers
        self.configs = configs

        self.device = torch.device("cuda") if self.use_cuda else torch.device("cpu")

    def fix_seed(self):
        seed = self.configs["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def snapshot_models(self, gen, dis, epoch):
        torch.save(
            gen.state_dict(), str(self.log_dir / "gen_{:05d}.pytorch".format(epoch))
        )
        torch.save(
            dis.state_dict(), str(self.log_dir / "dis_{:05d}.pytorch".format(epoch))
        )

    def train(self):
        # Retrieve models and move them if necessary
        gen, dis = self.models["gen"], self.models["dis"]
        if torch.cuda.is_available():
            gen.cuda()
            dis.cuda()

        # Retrieve optimizers
        opt_gen = self.optimizers["gen"]
        opt_dis = self.optimizers["dis"]

        # Initialize logger

        # Start training
        for i in range(self.configs["n_epochs"]):
            for x_real, c_true in iter(self.dataloader):
                batchsize = len(x_real)

                # --- phase generator ---
                gen.train()
                opt_gen.zero_grad()

                x_fake = gen.sample_images(batchsize)
                y_fake, c_fake = dis(x_fake.detach())

                loss_gen = dis.compute_adv_loss(y_fake, None)
                loss_gen += dis.compute_info_loss(c_fake, c_true)

                loss_gen.backward()
                opt_gen.step()

                # --- phase discriminator ---
                dis.train()
                opt_dis.zero_grad()

                y_real, c_real = dis(x_real)

                loss_dis = dis.compute_adv_loss(y_real, y_fake.detach())
                loss_dis += dis.compute_info_loss(c_real, c_true)

                loss_dis.backward()
                opt_dis.step()

                # log result
