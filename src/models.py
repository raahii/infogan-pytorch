from abc import ABCMeta
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import cuda

import utils

tensor = utils.new_tensor_module()


params = {"img_w": 28, "img_h": 28, "dim_z": 62, "dim_c": 12}


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class LatentVariable(object):
    def __init__(self, name: str, kind: str, prob: str, dim: int, **kwargs: Any):
        self.name: str = name
        self.type: str = kind
        self.dim: int = 0
        self.prob_name: str = prob

        # define probability distribution
        if prob == "normal":
            klass: Any = dist.normal.Normal(kwargs["min"], kwargs["max"])
            self.dim: int = dim
        elif prob == "uniform":
            klass = dist.uniform.Uniform(kwargs["min"], kwargs["max"])
            self.dim: int = dim
        elif prob == "categorical":
            klass = Categorical(kwargs["k"])
            self.dim: int = dim * kwargs["k"]

        self.prob: dist.Distribution = klass
        self.params = kwargs

    def __str__(self):
        return f"LatentVariable(name: {self.name}, type: {self.type}, prob: {self.prob_name}, dim: {self.dim})"


def build_latent_variables(lv_configs: Dict) -> List[LatentVariable]:
    lvars: List[LatentVariable] = []

    # first of all, add z variable
    count = 0
    for c in lv_configs:
        if c["kind"] == "z":
            if count > 1:
                raise Exception("More than two latent variables of type 'z' exist!")
            lvars.append(LatentVariable(**c))
            count += 1

    if count == 0:
        raise Exception("Latent variable of type 'z' doesn't exist!")

    # after that, add other variables
    for c in lv_configs:
        if c["kind"] != "z":
            lvars.append(LatentVariable(**c))

    return lvars


class Categorical:
    def __init__(self, k: int):
        from torch.distributions.categorical import Categorical as _Categorical

        self.k = k

        p = tensor.empty(k).fill_(1 / k)
        self.prob = _Categorical(p)

    def one_hot(self, x: torch.Tensor) -> torch.Tensor:
        b, c = tuple(x.shape)
        _x = torch.unsqueeze(x, 2)
        oh = tensor.empty(b, c, self.k).zero_()
        oh.scatter_(2, _x, 1)
        oh = oh.view(b, c * self.k)

        return oh

    def sample(self, shape: Sequence[int]) -> torch.Tensor:
        x = self.prob.sample(shape)
        x = self.one_hot(x)

        return x


class Generator(nn.Module):
    def __init__(self, latent_vars: List[LatentVariable]):
        super().__init__()

        self.latent_vars = latent_vars
        self.dim_input = sum(map(lambda x: x.dim, latent_vars))
        ngf = 64

        # main layers
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.dim_input, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )
        self.use_cuda = torch.cuda.is_available()
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)

    def forward_dummy(self) -> torch.Tensor:
        shape = (2, self.dim_input, 1, 1)
        dummy_tensor = tensor.empty(shape).normal_()
        return self.forward(dummy_tensor)

    def sample_z(self, batchsize: int) -> torch.Tensor:
        zs: List[torch.Tensor] = []
        for var in self.latent_vars:
            z = var.prob.sample([batchsize, var.dim])
            zs.append(z)

        lv = torch.cat(zs, dim=1)
        lv = lv.unsqueeze(-1).unsqueeze(-1)  # expand to image map

        if self.use_cuda:
            lv = lv.cuda()

        return lv

    def infer(self, n_samples: int) -> torch.Tensor:
        z = self.sample_z(n_samples)
        x = self.forward(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_vars: List[LatentVariable]):
        super().__init__()

        self.latent_vars = latent_vars
        self.dim_output = sum(map(lambda x: x.dim, latent_vars))
        ndf = 64

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.d_head: nn.Module = DHead(ndf)
        self.q_heads: List[nn.Module] = []  # TODO

        # setup output structure from latent variables configuration
        self.apply(weights_init)

    def forward(self, x):
        mid = self.main(x)

        y = self.d_head(mid)
        c: List[torch.Tensor] = []  # TODO: implement c vars

        return y, c

    def forward_dummy(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        shape = (2, 1, params["img_h"], params["img_w"])
        dummy_tensor = tensor.empty(shape).normal_()

        return self.forward(dummy_tensor)


class DHead(nn.Module):
    def __init__(self, ndf: int):
        super().__init__()
        self.main = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)


if __name__ == "__main__":
    from train import load_yaml

    configs = load_yaml("configs/default.yaml")
    latent_vars = build_latent_variables(configs["latent_variables"])

    g = Generator(latent_vars)
    x = g.infer(2)
    print(x.shape)

    d = Discriminator(latent_vars)
    y, c = d.forward(x)
    print(y.shape, len(c))
