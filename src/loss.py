import functools
import math
from typing import Dict, List

import torch
import torch.nn as nn

import utils
from models import LatentVariable

LABEL_REAL, LABEL_FAKE = 1, 0


class InfoGANLoss:
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        self.latent_vars = latent_vars

        self.discrete_loss = nn.CrossEntropyLoss()
        self.continuous_loss = NormalNLLLoss()
        self.device = utils.current_device()

    def __call__(
        self, cs_hat: Dict[str, List[torch.Tensor]], cs_true: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if cs_hat.keys() != cs_true.keys():
            raise Exception("The keys of cs_hat is different from cs_true")

        losses: List[torch.Tensor] = []
        for key in cs_hat.keys():
            c_hat: List[torch.Tensor] = cs_hat[key]
            c_true: torch.Tensor = cs_true[key]

            if self.latent_vars[key].prob_name == "categorical":
                _, targets = c_true.max(dim=1)
                losses.append(self.discrete_loss(c_hat[0], targets))
            elif self.latent_vars[key].prob_name == "uniform":
                losses.append(self.continuous_loss(c_true, c_hat[0], c_hat[1]))

        return functools.reduce(lambda x, y: x + y, losses)


class AdversarialLoss:
    def __init__(self):
        self.loss = nn.BCELoss()
        self.device = utils.current_device()

    def __call__(self, y_hat: torch.Tensor, label: int):
        if label not in [LABEL_REAL, LABEL_FAKE]:
            raise Exception("Invalid label is passed to adversarial loss")

        y_true = torch.full(y_hat.size(), label, device=self.device)
        return self.loss(y_hat, y_true)


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """

    def __call__(
        self, x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:

        eps = 1e-6
        nll = -0.5 * torch.log(torch.mul(var, 2 * math.pi) + eps)
        nll = nll - torch.pow(x - mu, 2)
        nll = torch.div(nll, (torch.mul(var, 2.0) + eps))
        nll = -torch.mean(torch.sum(nll, 1))

        return nll
