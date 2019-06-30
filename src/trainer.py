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
        configs: Dict,
        logger: Logger,
    ):

        self.dataloader = dataloader
        self.models = models
        self.optimizers = optimizers
        self.configs = configs
        self.latent_vars = models["gen"].latent_vars

        self.use_cuda = torch.cuda.is_available()
        self.device = utils.current_device()

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

    def dataset_samples(self, x: torch.Tensor, step: int):
        torchvision.utils.save_image(x, self.gen_img_path / f"data_{step}.jpg", nrow=5)

    def gen_random_images(self, gen: nn.Module):
        gen.eval()
        with torch.no_grad():
            zs = gen.sample_latent_vars(25)
            x = gen.infer(list(zs.values()))

        x = make_grid(x, 5, normalize=True)  # , scale_each=True)
        self.logger.tf_log_image(x, self.iteration, "random")
        torchvision.utils.save_image(
            x, self.gen_img_path / f"gen_random_{self.iteration}.jpg"
        )

    def gen_images_per_chars(self, gen: nn.Module):
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
            x, self.gen_img_path / f"gen_chars_{self.iteration}.jpg"
        )

    def train(self):
        # retrieve models and move them if necessary
        gen, dis = self.models["gen"], self.models["dis"]
        dhead, qhead = self.models["dhead"], self.models["qhead"]

        adv_loss = loss.AdversarialLoss()
        info_loss = loss.InfoGANLoss(self.latent_vars)

        if self.use_cuda:
            gen.cuda()
            dis.cuda()
            dhead.cuda()
            qhead.cuda()

        # retrieve optimizers
        opt_gen = self.optimizers["gen"]
        opt_dis = self.optimizers["dis"]

        # Define metrics
        self.logger.define("iteration", MetricType.Number)
        self.logger.define("epoch", MetricType.Number)
        self.logger.define("loss_gen", MetricType.Loss)
        self.logger.define("loss_dis", MetricType.Loss)
        self.logger.define("loss_q", MetricType.Loss)

        # Start training
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
                x_real = x_real.cuda() if self.use_cuda else x_real
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
                    self.gen_random_images(gen)
                    self.gen_images_per_chars(gen)

                # evaluate generated samples
                # if iteration % configs["evaluation_interval"] == 0:
                #    pass
