#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__all__ = [
    'SPoSE',
    'l1_regularization',
]

import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class SPoSE(nn.Module):

    def __init__(
        self,
        in_size: int,
        out_size: int,
        init_weights: bool = True,
    ):
        super(SPoSE, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(self.in_size, self.out_size, bias=False)

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def _initialize_weights(self) -> None:
        mean, std = .1, .01
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)


def l1_regularization(model, kwd) -> torch.Tensor:
    kwd_pattern = fr'{kwd}'
    l1_reg = torch.tensor(0., requires_grad=True)
    for n, p in model.named_parameters():
        if re.search(kwd_pattern, n):
            l1_reg = l1_reg + torch.norm(p, 1)
    return l1_reg


class IndependentLinearLayerWithoutIntercept(nn.Module):
    # can be extended with intercept if required, see commented part below
    def __init__(self, input_dim):
        super(IndependentLinearLayerWithoutIntercept, self).__init__()
        # Define a parameter for each input dimension for weights
        self.weights = nn.Parameter(torch.ones(input_dim))
        # Define a parameter for each input dimension for intercepts
        # self.intercepts = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # Scale each input dimension by its corresponding weight and add the intercept
        return x * self.weights  # + self.intercepts


class IndependentLinearLayerWithIntercept(nn.Module):
    # can be extended with intercept if required, see commented part below
    def __init__(self, input_dim):
        super(IndependentLinearLayerWithIntercept, self).__init__()
        # Define a parameter for each input dimension for weights
        self.weights = nn.Parameter(torch.ones(input_dim))
        # Define a parameter for each input dimension for intercepts
        self.intercepts = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        # Scale each input dimension by its corresponding weight and add the intercept
        return x * self.weights + self.intercepts


class SPoSE_ID(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_participants: int,
        init_weights: bool = True,
    ):

        super(SPoSE_ID, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(self.in_size, self.out_size, bias=False)
        self.individual_slopes = nn.Embedding(num_participants, self.out_size)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor, id: torch.Tensor) -> torch.Tensor:
        w_i = self.individual_slopes(id)
        return w_i * self.fc(x)

    def _initialize_weights(self) -> None:
        mean, std = .1, .01
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean, std)
