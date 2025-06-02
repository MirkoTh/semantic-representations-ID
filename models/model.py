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


def l1_regularization(model, kwd, agreement: str = "few") -> torch.Tensor:
    kwd_pattern = fr'{kwd}'
    l1_reg = torch.tensor(0., requires_grad=True)
    for n, p in model.named_parameters():
        if re.search(kwd_pattern, n):
            if agreement == "few":
                l1_reg = l1_reg + torch.norm(p, 1)
            elif agreement == "most":
                l1_reg = l1_reg + torch.norm(1-p, 1)
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
        mean_avg, std_avg = .1, .01
        mean_id, std_id = .5, .15
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean_avg, std_avg)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean_id, std_id)


class SPoSE_ID_Random(nn.Module):
    def __init__(self, in_size: int, out_size: int, num_participants: int,
        init_weights: bool = True,):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(in_size, out_size, bias=False)
        self.individual_slopes = nn.Embedding(num_participants, out_size)

        # Define learnable global mean & std
        self.global_mean = nn.Parameter(torch.ones(out_size))
        self.global_std = nn.Parameter(torch.ones(out_size))

        if init_weights:
            self._initialize_weights()

    def hierarchical_loss(self, id: torch.Tensor):
        """Encourage slopes to stay within a normal distribution."""
        return torch.mean((self.individual_slopes(id) - self.global_mean) ** 2 / (2 * self.global_std**2))

    def forward(self, x: torch.Tensor, id: torch.Tensor):
        w_i = self.individual_slopes(id)
        return w_i * self.fc(x)

    def _initialize_weights(self) -> None:
        mean_avg, std_avg = .1, .01
        mean_id, std_id = .5, .15
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean_avg, std_avg)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean_id, std_id)


class SPoSE_ID_IC(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_participants: int,
        init_weights: bool = True,
    ):

        super(SPoSE_ID_IC, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.fc = nn.Linear(self.in_size, self.out_size, bias=False)
        self.individual_slopes = nn.Embedding(num_participants, self.out_size)
        self.individual_intercepts = nn.Embedding(
            num_participants, self.out_size)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor, id: torch.Tensor) -> torch.Tensor:
        w_i = self.individual_slopes(id)
        ic_i = self.individual_intercepts(id)
        return ic_i + w_i * self.fc(x)

    def _initialize_weights(self) -> None:
        mean_avg, std_avg = .1, .01
        mean_id, std_id = .5, .15
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean_avg, std_avg)
            elif isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean_id, std_id)


class Weighted_Embedding(nn.Module):
    def __init__(
        self,
        embed_size: int,
        num_participants: int,
        init_weights: bool = True,
    ):
        super(Weighted_Embedding, self).__init__()
        self.embed_size = embed_size
        self.num_participants = num_participants
        self.individual_slopes = nn.Embedding(
            num_participants, self.embed_size)
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor, id: torch.Tensor) -> torch.Tensor:
        w_i = self.individual_slopes(id)
        return w_i * x

    def _initialize_weights(self) -> None:
        mean, std = .5, .01
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.normal_(mean, std)
