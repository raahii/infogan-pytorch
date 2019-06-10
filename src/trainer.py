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
        logger: Logger,
    ):

        self.dataloader = dataloader
        self.models = models
        self.optimizers = optimizers
        self.configs = configs

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.logger = logger

        self.iteration = 0
        self.epoch = 0

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

        # Start training
        for i in range(self.configs["n_epochs"]):
            self.epoch += 1
            for x_real, c_true in iter(self.dataloader):
                self.iteration += 1
                batchsize = len(x_real)

                # --- phase generator ---
                gen.train()
                opt_gen.zero_grad()

                x_fake = gen.infer(batchsize)
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

                # update metric
                self.logger.update("iteration", self.iteration)
                self.logger.update("epoch", self.epoch)
                self.logger.update("loss_gen", loss_gen.cpu().item())
                self.logger.update("loss_dis", loss_dis.cpu().item())

                # log
                if self.iteration % self.configs["log_interval"] == 0:
                    self.logger.log()
                    self.logger.tf_log()
                    self.logger.clear()

                # # snapshot models
                # if iteration % configs["snapshot_interval"] == 0:
                #     self.snapshot_models(sgen, cgen, idis, vdis, iteration)

                # # log samples
                # if iteration % configs["log_samples_interval"] == 0:
                #     self.generate_samples(sgen, cgen, iteration)

                # evaluate generated samples
                # if iteration % configs["evaluation_interval"] == 0:
                #    pass
