import abc
from collections import OrderedDict
from functools import reduce
from typing import Any, Dict, List, Sequence, Tuple, TypeVar, Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

import utils

tensor = utils.new_tensor_module()
FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]


params = {"img_w": 28, "img_h": 28, "dim_z": 62, "dim_c": 12}


def init_normal(layer):
    if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
        # print(layer)
        init.normal_(layer.weight.data, 0, 0.02)
    elif type(layer) in [nn.BatchNorm2d]:
        init.normal_(layer.weight.data, 1.0, 0.02)
        init.constant_(layer.bias.data, 0.0)


class LatentVariable(object):
    def __init__(self, name: str, kind: str, prob: str, dim: int, **kwargs: Any):
        self.name: str = name
        self.type: str = kind
        self.dim: int = 0
        self.prob_name: str = prob

        # define probability distribution
        if prob == "normal":
            klass = dist.normal.Normal(kwargs["min"], kwargs["max"])
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

        p = torch.FloatTensor(k).fill_(1 / k)
        self.prob = _Categorical(p)

    def one_hot(self, x: FloatTensor) -> FloatTensor:
        b, c = tuple(x.shape)
        _x = torch.unsqueeze(x, 2)
        oh = torch.FloatTensor(b, c, self.k).zero_()
        oh.scatter_(2, _x, 1)
        oh = oh.view(b, c * self.k)

        return oh

    def sample(self, shape: Sequence[int]) -> FloatTensor:
        x = self.prob.sample(shape)
        x = self.one_hot(x)

        return x


class Generator(nn.Module):
    def __init__(self, latent_vars: List[LatentVariable]):
        super().__init__()

        self.latent_vars = latent_vars
        self.dim_input = sum(map(lambda x: x.dim, latent_vars))

        # main layers
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.dim_input, 1024, 1, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 128, 7, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False),
            nn.Sigmoid(),
        )

        self.apply(init_normal)

    def forward(self, x: Variable) -> Variable:
        return self.main(x)

    def forward_dummy(self) -> Variable:
        shape = (2, self.dim_input, 1, 1)
        dummy_tensor = Variable(tensor.FloatTensor(*shape).normal_())
        return self.forward(dummy_tensor)

    def sample_z(self, batchsize: int) -> Variable:
        zs: List[FloatTensor] = []
        for var in self.latent_vars:
            z = var.prob.sample([batchsize, var.dim])
            zs.append(z)

        lv: FloatTensor = torch.cat(zs, dim=1)
        lv = lv.unsqueeze(-1).unsqueeze(-1)  # expand to image map

        return Variable(lv)

    def infer(self, n_samples: int) -> Variable:
        z = self.sample_z(n_samples)
        x = self.forward(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_vars: List[LatentVariable]):
        super().__init__()

        self.latent_vars = latent_vars
        self.dim_output = sum(map(lambda x: x.dim, latent_vars))

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
        )

        self.d_head: nn.Module = DHead()
        self.q_heads: List[nn.Module] = []  # TODO

        # setup output structure from latent variables configuration
        self.apply(init_normal)

    def forward(self, x: Variable) -> Tuple[Variable, List[Variable]]:
        mid = self.main(x)

        y = self.d_head(mid)
        c: List[Variable] = []  # TODO: implement c vars

        return y, c

    def forward_dummy(self) -> Tuple[Variable, List[Variable]]:
        shape = (2, 1, params["img_h"], params["img_w"])
        dummy_tensor = Variable(tensor.FloatTensor(*shape).normal_())

        return self.forward(dummy_tensor)


class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x):
        output = torch.sigmoid(self.conv(x))
        return output


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
