from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

tensor = torch.cuda if torch.cuda.is_available() else torch

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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.dim_input = params["dim_z"] + params["dim_c"]

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
    g = Generator()
    x = g.forward_dummy()
    print(x.shape)

    d = Discriminator()
    y, c = d.forward_dummy()
    print(y.shape, len(c))
