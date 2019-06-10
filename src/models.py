import abc
from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
import torch.distributions as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

tensor = torch.cuda if torch.cuda.is_available() else torch
FloatTensor = Union[torch.FloatTensor, torch.cuda.FloatTensor]


params = {"img_w": 28, "img_h": 28, "dim_z": 62, "dim_c": 12}


def init_normal(layer):
    if type(layer) in [nn.Conv2d, nn.ConvTranspose2d]:
        # print(layer)
        init.normal_(layer.weight.data, 0, 0.02)
    elif type(layer) in [nn.BatchNorm2d]:
        init.normal_(layer.weight.data, 1.0, 0.02)
        init.constant_(layer.bias.data, 0.0)


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(
            var.mul(2.0) + 1e-6
        )
        nll = -(logli.sum(1).mean())

        return nll


class Samplable(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sample(self, shape: List[int]):
        pass


class Categorical(object):
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

    def sample(self, shape: Iterable[int]) -> FloatTensor:
        x = self.prob.sample(shape)
        x = self.one_hot(x)

        return x


class Generator(nn.Module):
    def __init__(self, lv_config):
        super().__init__()

        # setup latent variables
        self.latent_vars: OrderedDict[str, Tuple[Samplable, int]] = OrderedDict()
        self.dim_input: int = 0
        for name, conf in lv_config.items():
            dim = conf["dim"]

            # define probability class
            if conf["type"] == "normal":
                klass = dist.normal.Normal(conf["min"], conf["max"])
                self.dim_input += dim
            elif conf["type"] == "uniform":
                klass = dist.uniform.Uniform(conf["min"], conf["max"])
                self.dim_input += dim
            elif conf["type"] == "categorical":
                klass = Categorical(conf["k"])
                self.dim_input += dim * conf["k"]  # one hot vector

            self.latent_vars[name] = (klass, dim)

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
        variables: List[FloatTensor] = []
        for (prob, dim) in self.latent_vars.values():
            var = prob.sample([batchsize, dim])
            variables.append(var)

        lv = torch.cat(variables, dim=1)
        lv = lv.unsqueeze(-1).unsqueeze(-1)  # expand to image map

        return Variable(lv)

    def infer(self, n_samples: int) -> Variable:
        z = self.sample_z(n_samples)
        x = self.forward(z)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

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

        self.dhead = DHead()
        self.qhead = QHead()
        self.apply(init_normal)

        self.loss_bce = nn.BCELoss(reduction="mean")
        self.loss_dis = nn.CrossEntropyLoss()
        self.loss_con = NormalNLLLoss

        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def forward(self, x: Variable) -> Variable:
        _inter = self.main(x)
        y = self.dhead(_inter)
        c = self.qhead(_inter)

        return y, c

    def forward_dummy(self) -> List[Variable]:
        shape = (2, 1, params["img_h"], params["img_w"])
        dummy_tensor = Variable(tensor.FloatTensor(*shape).normal_())

        return self.forward(dummy_tensor)

    def compute_adv_loss(self, y_real: Variable, y_fake: Variable) -> Variable:
        ones = torch.ones_like(y_real, device=self.device)
        zeros = torch.zeros_like(y_fake, device=self.device)

        loss = self.loss_bce(y_real, ones)
        loss += self.loss_bce(y_fake, zeros)

        return loss

    def compute_info_loss(self, c_true: Variable, c_hat: Variable) -> Variable:
        return Variable
        pass


class DHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1024, 1, 1)

    def forward(self, x: Variable) -> Variable:
        output = torch.sigmoid(self.conv(x))

        return output


class QHead(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1024, 128, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)

        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)

    def forward(self, x: Variable) -> Variable:
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.1, inplace=True)

        logits = self.conv_disc(x).squeeze()

        mu = self.conv_mu(x).squeeze()
        var = torch.exp(self.conv_var(x).squeeze())

        return logits, mu, var


if __name__ == "__main__":
    from train import load_yaml

    configs = load_yaml("configs/default.yaml")
    lv_config = configs["latent_variables"]
    # print(lv_config)

    g = Generator(lv_config)
    print(g.infer(10).shape)
    # x = g.forward_dummy()
    # print(x.shape)
    #
    # d = Discriminator()
    # y, c = d.forward_dummy()
    # print(y.shape, len(c))
