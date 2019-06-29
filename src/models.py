from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch import cuda

import utils

tensor = utils.new_tensor_module()


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
        self.kind: str = kind
        self.dim: int = dim
        self.prob_name: str = prob

        # define probability distribution
        if prob == "normal":
            klass: Any = dist.normal.Normal(kwargs["mu"], kwargs["var"])
            self.cdim: int = dim
        elif prob == "uniform":
            klass = dist.uniform.Uniform(kwargs["min"], kwargs["max"])
            self.cdim: int = dim
        elif prob == "categorical":
            klass = Categorical(kwargs["k"])
            self.cdim: int = dim * kwargs["k"]

        self.prob: dist.Distribution = klass
        self.params = kwargs

    def __str__(self):
        return f"<LatentVariable(name: {self.name}, kind: {self.kind}, prob: {self.prob_name}, dim: {self.dim})>"

    def __repr__(self):
        return str(self)


def build_latent_variables(lv_configs: Dict) -> Dict[str, LatentVariable]:
    lvars: OrderedDict[str, LatentVariable] = OrderedDict()

    # first of all, add z variable
    count = 0
    for c in lv_configs:
        if c["kind"] == "z":
            if count > 1:
                raise Exception("More than two latent variables of kind 'z' exist!")
            lvars[c["name"]] = LatentVariable(**c)
            count += 1

    if count == 0:
        raise Exception("Latent variable of kind 'z' doesn't exist!")

    # after that, add other variables
    for c in lv_configs:
        if c["kind"] == "z":
            continue

        if c["name"] in lvars:
            raise Exception("Latent variable name is not unique.")

        lvars[c["name"]] = LatentVariable(**c)

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
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        super().__init__()

        self.latent_vars = latent_vars
        self.dim_input = sum(map(lambda x: x.cdim, latent_vars.values()))
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

    def sample_latent_vars(self, batchsize: int) -> Dict[str, torch.Tensor]:
        zs: OrderedDict[str, torch.Tensor] = OrderedDict()

        for name, var in self.latent_vars.items():
            zs[name] = var.prob.sample([batchsize, var.dim])

        return zs

    def infer(self, zs: List[torch.Tensor]) -> torch.Tensor:
        z = torch.cat(zs, dim=1)
        z = z.unsqueeze(-1).unsqueeze(-1)  # expand to image map

        if self.use_cuda:
            z = z.cuda()

        x = self.forward(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        super().__init__()

        self.latent_vars = latent_vars
        self.dim_output = sum(map(lambda x: x.cdim, latent_vars.values()))
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
            # state size. (ndf*8) x 4 x 4
        )

        # setup output structure from latent variables configuration
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)

    def forward_dummy(self) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        shape = (2, 1, 64, 64)
        dummy_tensor = tensor.empty(shape).normal_()

        return self.forward(dummy_tensor)


class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 64
        self.main = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # state size. 1 x 1 x 1
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)


class QHead(nn.Module):
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        super().__init__()

        ndf = 64
        self.main = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 2, 4, 1, 0, bias=False)
            # state size. 128 x 2 x 2
        )

        self.convs: Dict[str, List[nn.Conv2d]] = {}
        for name, var in latent_vars.items():
            if var.kind == "z":
                continue

            if var.prob_name == "normal":
                # normal distribution: estimate mu and var
                convs = [
                    nn.Conv2d(ndf * 2, var.cdim, 1),
                    nn.Conv2d(ndf * 2, var.cdim, 1),
                ]
            else:
                # categorical: estimate a scaler class value
                convs = [nn.Conv2d(ndf * 2, var.cdim, 1)]

            for i, conv in enumerate(convs):
                conv.apply(weights_init)
                setattr(self, f"conv_{name}_{i}", conv)

            self.convs[name] = convs

    def forward(self, x):
        mid = self.main(x)

        y: Dict[str, torch.Tensor] = {}
        for name, convs in self.convs.items():
            y[name] = [conv(mid) for conv in convs]

        return y


if __name__ == "__main__":
    from train import load_yaml

    configs = load_yaml("configs/debug.yaml")
    latent_vars = build_latent_variables(configs["latent_variables"])

    g = Generator(latent_vars)
    zs = g.sample_latent_vars(2)
    for k, v in zs.items():
        print(k, v.shape)
    x = g.infer(list(zs.values()))
    print(x.shape)

    d = Discriminator(latent_vars)
    d_head, q_head = DHead(), QHead(latent_vars)

    mid = d(x)
    y, c = d_head(mid), q_head(mid)

    print(y.shape, list(map(lambda x: x.size(), c.values())))
