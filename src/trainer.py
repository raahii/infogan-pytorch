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
import util
from logger import Logger, MetricType
from variable import LatentVariable


class Trainer(object):
    def __init__(
        self,
        dataloader: DataLoader,
        latent_vars: Dict[str, LatentVariable],
        models: Dict[str, nn.Module],
        optimizers: Dict[str, Any],
        losses: Dict[str, Any],
        configs: Dict[str, Any],
        logger: Logger,
    ):

        self.dataloader = dataloader
        self.latent_vars = latent_vars
        self.models = models
        self.optimizers = optimizers
        self.losses = losses
        self.configs = configs
        self.logger = logger

        self.device = util.current_device()

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
        torch.backends.cudnn.benchmark = True

    def snapshot_models(self):
        for name, model in self.models.items():
            torch.save(
                model.state_dict(),
                str(self.model_snapshots_path / f"{name}_{self.iteration:05d}.pytorch"),
            )

    def gen_random_images(self, gen: nn.Module):
        gen.eval()
        with torch.no_grad():
            zs = gen.module.sample_latent_vars(100)
            x = gen.module.infer(list(zs.values()))

        x = make_grid(x, 10, normalize=True)  # , scale_each=True)
        self.logger.tf_log_image(x, self.iteration, "random")
        torchvision.utils.save_image(
            x, self.gen_images_path / f"random_{self.iteration}.jpg"
        )

    def gen_images_discrete(self, gen: nn.Module, var_name: str):
        """
        Generate images with varying discrete variable
        """

        k: int = self.latent_vars[var_name].params["k"]

        gen.eval()
        with torch.no_grad():
            zs = gen.module.sample_latent_vars(100)

            # overwrite variable(e.g. c1) to intentional values
            idx = np.arange(k).repeat(k)
            one_hot = np.zeros((k * k, k))
            one_hot[range(k * k), idx] = 1
            zs[var_name] = torch.tensor(one_hot, device=self.device, dtype=torch.float)

            x = gen.module.infer(list(zs.values()))

        x = make_grid(x, k, normalize=True)
        self.logger.tf_log_image(x, self.iteration, var_name)
        torchvision.utils.save_image(
            x, self.gen_images_path / f"{var_name}_{self.iteration}.jpg"
        )

    def gen_images_continuous(
        self, gen: nn.Module, var_name_dis: str, var_name_con: str
    ):
        """
        Generate images with varying continuous variable
        """

        k: int = self.latent_vars[var_name_dis].params["k"]
        # _min: int = self.latent_vars[var_name_con].params["min"]
        # _max: int = self.latent_vars[var_name_con].params["max"]
        _min: int = -2
        _max: int = 2

        gen.eval()
        with torch.no_grad():
            zs = gen.module.sample_latent_vars(k)

            for var_name in self.latent_vars.keys():

                if var_name == var_name_dis:
                    # overwrite discrete variable to intentional values
                    idx = np.arange(k).repeat(k)  # equal along row direction
                    one_hot = np.zeros((k * k, k))
                    one_hot[range(k * k), idx] = 1
                    zs[var_name_dis] = torch.tensor(
                        one_hot, device=self.device, dtype=torch.float
                    )
                elif var_name == var_name_con:
                    # overwrite continuous variable to intentional values
                    interp = np.linspace(_min, _max, k)
                    interp = np.expand_dims(interp, 1)
                    interp = np.tile(interp, (k, 1))  # equal along col direction
                    zs[var_name_con] = torch.tensor(
                        interp, device=self.device, dtype=torch.float
                    )
                else:
                    zs[var_name] = (
                        zs[var_name].view(k, 1, -1).repeat(1, k, 1).view(k * k, -1)
                    )
            x = gen.module.infer(list(zs.values()))

        x = make_grid(x, k, normalize=True)
        self.logger.tf_log_image(x, self.iteration, f"{var_name_dis}_{var_name_con}")
        torchvision.utils.save_image(
            x,
            self.gen_images_path
            / f"{var_name_dis}_{var_name_con}_{self.iteration}.jpg",
        )

    def train(self):
        # retrieve models and move them if necessary
        gen, dis = self.models["gen"], self.models["dis"]
        dhead, qhead = self.models["dhead"], self.models["qhead"]

        n_gpus = torch.cuda.device_count()
        if n_gpus >= 1:
            gen = nn.DataParallel(gen)
            dis = nn.DataParallel(dis)
            dhead = nn.DataParallel(dhead)
            qhead = nn.DataParallel(qhead)

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

        for k, v in self.latent_vars.items():
            if v.kind == "z":
                continue
            self.logger.define(f"loss_{k}", MetricType.Loss)

        # start training
        self.logger.info(f"Start training, device: {self.device} n_gpus: {n_gpus}")
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
                zs = gen.module.sample_latent_vars(batchsize)
                x_fake = gen.module.infer(list(zs.values()))
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

                # compute mutual information loss
                c_true = {k: zs[k] for k in c_fake.keys()}
                loss_q, loss_q_details = info_loss(c_fake, c_true)
                loss_gen += loss_q

                loss_gen.backward()
                opt_gen.step()

                # update metrics
                self.logger.update("iteration", self.iteration)
                self.logger.update("epoch", self.epoch)
                self.logger.update("loss_gen", loss_gen.cpu().item())
                self.logger.update("loss_dis", loss_dis.cpu().item())
                self.logger.update("loss_q", loss_q.cpu().item())

                for k, v in loss_q_details.items():
                    self.logger.update(f"loss_{k}", v.cpu().item())

                # log metrics
                if self.iteration % self.configs["log_interval"] == 0:
                    self.logger.log()
                    self.logger.log_tensorboard("iteration")
                    self.logger.clear()

                # snapshot models
                if self.iteration % self.configs["snapshot_interval"] == 0:
                    self.snapshot_models()

                # generate and save samples
                if self.iteration % self.configs["log_samples_interval"] == 0:
                    self.gen_random_images(gen)
                    self.gen_images_discrete(gen, "c1")
                    self.gen_images_continuous(gen, "c1", "c2")
                    self.gen_images_continuous(gen, "c1", "c3")
                    # for name, var in self.latent_vars.items():
                    #     if var.prob_name == "categorical":
                    #         self.gen_images_discrete(gen, name)
                    #     else:
                    #         self.gen_images_continuous(gen, "c1", name)
