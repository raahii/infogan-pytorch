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
import utils
from logger import Logger, MetricType


class Trainer(object):
    def __init__(
        self,
        dataloader: DataLoader,
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Any],
        losses: Dict[str, Any],
        configs: Dict[str, Any],
        logger: Logger,
    ):

        self.dataloader = dataloader
        self.models = models
        self.optimizers = optimizers
        self.losses = losses
        self.configs = configs
        self.logger = logger

        self.device = utils.current_device()

        self.gen_images_path = self.logger.path / "images"
        self.model_snapshots_path = self.logger.path / "models"
        for p in [self.gen_images_path, self.model_snapshots_path]:
            p.mkdir(parents=True, exist_ok=True)

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

    def snapshot_models(self):
        for name, model in self.models.items():
            torch.save(
                model.state_dict(),
                str(self.model_snapshots_path / f"{name}_{self.iteration:05d}.pytorch"),
            )

    def gen_random_images(self, gen: nn.Module):
        gen.eval()
        with torch.no_grad():
            zs = gen.sample_latent_vars(25)
            x = gen.infer(list(zs.values()))

        x = make_grid(x, 5, normalize=True)  # , scale_each=True)
        self.logger.tf_log_image(x, self.iteration, "random")
        torchvision.utils.save_image(
            x, self.gen_images_path / f"random_{self.iteration}.jpg"
        )

    def gen_images_per_char(self, gen: nn.Module):
        gen.eval()
        with torch.no_grad():
            zs = gen.sample_latent_vars(100)
            idx = np.arange(10).repeat(10)
            one_hot = np.zeros((100, 10))
            one_hot[range(100), idx] = 1
            zs["c1"] = torch.tensor(one_hot, device=self.device, dtype=torch.float)
            x = gen.infer(list(zs.values()))

        x = make_grid(x, 10, normalize=True)
        self.logger.tf_log_image(x, self.iteration, "chars")
        torchvision.utils.save_image(
            x, self.gen_images_path / f"chars_{self.iteration}.jpg"
        )

    def train(self):
        # retrieve models and move them if necessary
        gen, dis = self.models["gen"], self.models["dis"]
        dhead, qhead = self.models["dhead"], self.models["qhead"]
        gen, dis = gen.to(self.device), dis.to(self.device)
        dhead, qhead = dhead.to(self.device), qhead.to(self.device)

        # optimizers
        opt_gen = self.optimizers["gen"]
        opt_dis = self.optimizers["dis"]

        # losses
        adv_loss = self.losses["adv"]
        info_loss = self.losses["info"]

        # define metrics
        self.logger.define("iteration", MetricType.Number)
        self.logger.define("epoch", MetricType.Number)
        self.logger.define("loss_gen", MetricType.Loss)
        self.logger.define("loss_dis", MetricType.Loss)
        self.logger.define("loss_q", MetricType.Loss)

        # start training
        self.logger.info(f"Start training, device: {self.device}")
        self.logger.print_header()
        for i in range(self.configs["n_epochs"]):
            self.epoch += 1
            for x_real, _ in iter(self.dataloader):
                self.iteration += 1
                batchsize = len(x_real)

                gen.train()
                dis.train()
                dhead.train()
                qhead.train()

                # ------- discrminator phase -------
                opt_dis.zero_grad()

                # train with real
                x_real = x_real.to(self.device)
                y_real = dhead(dis(x_real))
                loss_dis_real = adv_loss(y_real, loss.LABEL_REAL)
                loss_dis_real.backward()

                # train with fake
                zs = gen.sample_latent_vars(batchsize)
                x_fake = gen.infer(list(zs.values()))
                y_fake = dhead(dis(x_fake.detach()))
                loss_dis_fake = adv_loss(y_fake, loss.LABEL_FAKE)

                loss_dis_fake.backward()
                opt_dis.step()

                loss_dis = loss_dis_real + loss_dis_fake

                # ------- generator phase -------
                opt_gen.zero_grad()

                mid = dis(x_fake)
                y_fake, c_fake = dhead(mid), qhead(mid)

                # compute loss as fake samples are real
                loss_gen = adv_loss(y_fake, loss.LABEL_REAL)

                # compute infogan loss
                c_true = {k: zs[k] for k in c_fake.keys()}
                loss_q = info_loss(c_fake, c_true)
                loss_gen += loss_q

                loss_gen.backward()
                opt_gen.step()

                # update metric
                self.logger.update("iteration", self.iteration)
                self.logger.update("epoch", self.epoch)
                self.logger.update("loss_gen", loss_gen.cpu().item())
                self.logger.update("loss_dis", loss_dis.cpu().item())
                self.logger.update("loss_q", loss_q.cpu().item())

                # log metrics
                if self.iteration % self.configs["log_interval"] == 0:
                    self.logger.log()
                    self.logger.log_tensorboard("iteration")
                    # self.logger.clear()

                # snapshot models
                if self.iteration % self.configs["snapshot_interval"] == 0:
                    self.snapshot_models()

                # generate and save samples
                if self.iteration % self.configs["log_samples_interval"] == 0:
                    self.gen_random_images(gen)
                    self.gen_images_per_char(gen)
