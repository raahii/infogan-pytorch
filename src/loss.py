from typing import Any, Dict, List

import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import utils

tensor = utils.new_tensor_module()

LABEL_REAL, LABEL_FAKE = 1, 0


class InfoGANLoss:
    def __init__(self, lv_config: Dict):
        self.nnl_loss = NormalNLLLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # parse config for latent variales
        losses: List[Any] = []
        for conf in lv_config:
            if conf["kind"] == "z":
                continue

            # discrete probability distribution
            if conf["prob"] in ["categorical"]:
                losses.append(self.ce_loss)
            elif conf["prob"] in ["normal"]:
                losses.append(self.ce_loss)
            else:
                raise Exception(
                    "Invalid ditribution is defined in configuration of latent variable"
                )

        self.losses = losses

    def __call__(self, cs_hat: List[Variable], cs_true: List[Variable]):
        if len(cs_hat) != len(cs_true):
            raise Exception(
                "Number of variables between 'c_hat' and 'c_true' is mismatch!"
                + f"expected: {len(cs_hat)}"
                + f"actual: {len(cs_true)}"
            )

        if len(cs_true) != self.losses:
            raise Exception(
                "Number of variables 'c' is mismatch!"
                + f"expected: {len(self.losses)}"
                + f"actual: {len(cs_true)}"
            )
        pass


class AdversarialLoss:
    def __init__(self):
        self.loss = nn.BCELoss()

    def __call__(self, y_hat: Variable, label: int):
        if label not in [LABEL_REAL, LABEL_FAKE]:
            raise Exception("Invalid label is passed to adversarial loss")

        y_true = tensor.FloatTensor(y_hat.size()).fill_(label)
        return self.loss(y_hat, y_true)


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
