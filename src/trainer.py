import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

import loss
from logger import Logger, MetricType


class Trainer(object):
    def __init__(
        self,
        dataloader: DataLoader,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Any],
        configs: Dict,
        logger: Logger,
    ):

        self.dataloader = dataloader
        self.models = models
        self.optimizers = optimizers
        self.configs = configs

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda") if self.use_cuda else torch.device("cpu")

        self.logger = logger
        self.gen_img_path = Path(configs["gen_img_path"])

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

    def generate_samples(self, gen: nn.Module, step: int):
        x = x.repeat(1, 3, 1, 1)
        # x_grid = torchvision.utils.make_grid(x, 5, normalize=True)
        # self.logger.tf_log_image(x_grid, step, "x_fake")
        # print("samples")

    def dataset_samples(self, x: torch.Tensor, step: int):
        torchvision.utils.save_image(x, self.gen_img_path / f"data_{step}.jpg", nrow=5)

    def train(self):
        # retrieve models and move them if necessary
        gen, dis = self.models["gen"], self.models["dis"]
        if self.use_cuda:
            gen.cuda()
            dis.cuda()

        # retrieve optimizers
        opt_gen = self.optimizers["gen"]
        opt_dis = self.optimizers["dis"]

        adv_loss = loss.AdversarialLoss()
        # info_loss = loss.InfoGANLoss(self.configs["latent_variables"])

        # Define metrics
        self.logger.define("iteration", MetricType.Number)
        self.logger.define("epoch", MetricType.Number)
        self.logger.define("loss_gen", MetricType.Loss)
        self.logger.define("loss_dis", MetricType.Loss)

        # Start training
        self.logger.info(f"Start training, device: {self.device}")
        self.logger.print_header()
        for i in range(self.configs["n_epochs"]):
            self.epoch += 1
            for x_real, c_true in iter(self.dataloader):
                self.iteration += 1
                batchsize = len(x_real)

                gen.train()
                dis.train()

                ############################
                # Update Discriminator:
                #  maximize log(D(x)) + log(1 - D(G(z)))
                ############################
                opt_dis.zero_grad()

                # train with real
                x_real = x_real.cuda() if self.use_cuda else x_real
                y_real, c_real = dis(x_real)
                loss_dis_real = adv_loss(y_real, loss.LABEL_REAL)

                loss_dis_real.backward()

                # train with fake
                x_fake = gen.infer(batchsize)
                y_fake, c_fake = dis(x_fake.detach())
                loss_dis_fake = adv_loss(y_fake, loss.LABEL_FAKE)

                loss_dis_fake.backward()
                opt_dis.step()

                loss_dis = loss_dis_real + loss_dis_fake

                ############################
                # Update Generator:
                #  maximize log(D(G(z)))
                ###########################
                opt_gen.zero_grad()

                y_fake, c_fake = dis(x_fake)

                # compute loss as fake samples are real
                loss_gen = adv_loss(y_fake, loss.LABEL_REAL)
                # loss_gen += info_loss(c_fake, c_true)

                loss_gen.backward()
                opt_gen.step()

                # update metric
                self.logger.update("iteration", self.iteration)
                self.logger.update("epoch", self.epoch)
                self.logger.update("loss_gen", loss_gen.cpu().item())
                self.logger.update("loss_dis", loss_dis.cpu().item())

                # log
                if self.iteration % self.configs["log_interval"] == 0:
                    self.logger.log()
                    self.logger.log_tensorboard("iteration")
                    # self.logger.clear()

                # # snapshot models
                # if iteration % configs["snapshot_interval"] == 0:
                #     self.snapshot_models(sgen, cgen, idis, vdis, iteration)

                # log samples
                if self.iteration % self.configs["log_samples_interval"] == 0:
                    gen.eval()
                    with torch.no_grad():
                        x = gen.infer(25)

                    x = make_grid(x, 5, normalize=True, scale_each=True)
                    self.logger.tf_log_image(x, self.iteration, "x_fake")
                    torchvision.utils.save_image(
                        x, self.gen_img_path / f"gen_{self.iteration}.jpg"
                    )

                # evaluate generated samples
                # if iteration % configs["evaluation_interval"] == 0:
                #    pass
