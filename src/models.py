from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn

import utils


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Noise(nn.Module):
    def __init__(self, use_noise: float, sigma: float = 0.2):
        super().__init__()
        self.use_noise = use_noise
        self.sigma = sigma
        self.device = utils.current_device()

    def forward(self, x):
        if self.use_noise:
            return (
                x
                + self.sigma
                * torch.empty(
                    x.size(), device=self.device, requires_grad=False
                ).normal_()
            )
        return x


class LatentVariable(object):
    def __init__(self, name: str, kind: str, prob: str, dim: int, **kwargs: Any):
        self.name: str = name
        self.kind: str = kind
        self.dim: int = dim
        self.cdim: int = dim
        self.prob_name: str = prob

        # define probability distribution
        if prob == "normal":
            klass: Any = dist.normal.Normal(kwargs["mu"], kwargs["var"])
        elif prob == "uniform":
            klass = dist.uniform.Uniform(kwargs["min"], kwargs["max"])
        elif prob == "categorical":
            klass = Categorical(kwargs["k"])
            self.cdim = dim * kwargs["k"]
            # k = kwargs["k"]
            # p = torch.full((k,), 1.0 / k)
            # klass = dist.categorical.Categorical(p)

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

        p = torch.empty(k).fill_(1 / k)
        self.prob = _Categorical(p)

    def one_hot(self, x: torch.Tensor) -> torch.Tensor:
        b, c = tuple(x.shape)
        _x = torch.unsqueeze(x, 2)
        oh = torch.empty(b, c, self.k).zero_()
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
        self.device = utils.current_device()

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
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)

    def forward_dummy(self) -> torch.Tensor:
        shape = (2, self.dim_input, 1, 1)
        dummy_tensor = torch.empty(shape, device=self.device).normal_()
        return self.forward(dummy_tensor)

    def sample_latent_vars(self, batchsize: int) -> Dict[str, torch.Tensor]:
        zs: OrderedDict[str, torch.Tensor] = OrderedDict()

        for name, var in self.latent_vars.items():
            zs[name] = var.prob.sample([batchsize, var.dim])
            zs[name] = zs[name].to(self.device)

        return zs

    def infer(self, zs: List[torch.Tensor]) -> torch.Tensor:
        z = torch.cat(zs, dim=1)
        z = z.unsqueeze(-1).unsqueeze(-1)  # expand to image map

        x = self.forward(z)
        return x

    @property
    def module(self) -> nn.Module:
        return self


class Discriminator(nn.Module):
    def __init__(self, configs: Dict[str, Any]):
        super().__init__()

        self.configs = configs

        ndf = 64
        self.device = utils.current_device()

        use_noise: bool = configs["use_noise"]
        noise_sigma: float = configs["noise_sigma"]

        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(1, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            Noise(use_noise, sigma=noise_sigma),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            Noise(use_noise, sigma=noise_sigma),
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
        dummy_tensor = torch.empty(shape, device=self.device).normal_()

        return self.forward(dummy_tensor)


class DHead(nn.Module):
    def __init__(self):
        super().__init__()
        ndf = 64
        self.main = nn.Sequential(
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 3, 1, 1, bias=False),
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
            nn.Conv2d(ndf * 8, ndf * 2, 4, 1, 0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # state size. 128 x 2 x 2
        )

        # generate each head module from latent variable
        self.convs: nn.ModuleDict = nn.ModuleDict()
        for name, var in latent_vars.items():
            if var.kind == "z":
                continue

            if var.prob_name == "uniform":
                self.convs[name] = nn.Sequential(
                    nn.Conv2d(ndf * 2, var.cdim, 1), nn.Tanh()
                )
            else:
                self.convs[name] = nn.Conv2d(ndf * 2, var.cdim, 1)

        self.apply(weights_init)

    def forward(self, x):
        mid = self.main(x)

        ys: Dict[str, torch.Tensor] = {}
        for name, conv in self.convs.items():
            ys[name] = conv(mid).squeeze()

        return ys


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

    d = Discriminator(configs["models"]["dis"])
    d_head, q_head = DHead(), QHead(latent_vars)

    mid = d(x)
    y, c = d_head(mid), q_head(mid)

    print(mid.shape, y.shape, list(map(lambda x: [_x.size() for _x in x], c.values())))
