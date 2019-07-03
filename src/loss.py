import functools
import math
from typing import Dict, List

import torch
import torch.nn as nn

import util
from variable import LatentVariable

LABEL_REAL, LABEL_FAKE = 1, 0


class AdversarialLoss:
    def __init__(self):
        self.loss = nn.BCELoss(reduction="mean")
        self.device = util.current_device()

    def __call__(self, y_hat: torch.Tensor, label: int):
        if label not in [LABEL_REAL, LABEL_FAKE]:
            raise Exception("Invalid label is passed to adversarial loss")

        y_true = torch.full(y_hat.size(), label, device=self.device)
        return self.loss(y_hat, y_true)


class InfoGANLoss:
    def __init__(self, latent_vars: Dict[str, LatentVariable]):
        self.latent_vars = latent_vars

        self.discrete_loss = nn.CrossEntropyLoss()
        self.continuous_loss = NormalNLLLoss()
        self.device = util.current_device()

    def __call__(
        self, cs_hat: Dict[str, torch.Tensor], cs_true: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if cs_hat.keys() != cs_true.keys():
            raise Exception("The keys of cs_hat is different from cs_true")

        losses: List[torch.Tensor] = []
        for key in cs_hat.keys():
            c_hat, c_true = cs_hat[key], cs_true[key]

            if self.latent_vars[key].prob_name == "categorical":
                _, targets = c_true.max(dim=1)
                loss = self.discrete_loss(c_hat, targets)
            elif self.latent_vars[key].prob_name == "uniform":
                ln_var = torch.ones(c_hat.size(), device=self.device)
                loss = self.continuous_loss(c_true.squeeze(), c_hat, ln_var)

            loss *= self.latent_vars[key].params["weight"]
            losses.append(loss)

        return functools.reduce(lambda x, y: x + y, losses)


class NormalNLLLoss:
    def __call__(
        self, x: torch.Tensor, mean: torch.Tensor, ln_var: torch.Tensor
    ) -> torch.Tensor:

        eps = 1e-6
        nll = -0.5 * torch.pow(x - mean, 2) * torch.exp(-ln_var)
        nll = (ln_var + math.log(2 * math.pi + eps)) / 2 - nll

        return torch.mean(nll)
