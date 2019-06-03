from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

tt = torch.cuda if torch.cuda.is_available() else torch

params = {"img_w": 28, "img_h": 28, "dim_z": 62, "dim_c": 12}


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

    def forward(self, x: Variable) -> Variable:
        return self.main(x)

    def forward_dummy(self) -> Variable:
        shape = (2, self.dim_input, 1, 1)
        dummy_tensor = Variable(tt.FloatTensor(*shape).normal_())
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

    def forward(self, x: Variable) -> Variable:
        return self.main(x)

    def forward_dummy(self) -> List[Variable]:
        shape = (2, 1, params["img_h"], params["img_w"])
        dummy_tensor = Variable(tt.FloatTensor(*shape).normal_())
        _inter = self.forward(dummy_tensor)
        y = self.dhead(_inter)
        c1, c2, c3 = self.qhead(_inter)

        return y, c1, c2, c3


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
    # import yaml
    #
    # f = open("./configs/mnist.yaml")
    # params = yaml.load(f)
    # print(params)
    # f.close()

    g = Generator()
    x = g.forward_dummy()
    print(x.shape)

    d = Discriminator()
    y, c1, c2, c3 = d.forward_dummy()
    print(y.shape, c1.shape, c2.shape, c3.shape)
