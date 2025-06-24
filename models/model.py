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
import utils as ut


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

class Scaling_ID(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_participants: int,
        init_weights: bool = True,
    ):
        super(Scaling_ID, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.individual_temps = nn.Embedding(num_participants, 1)

        if init_weights:
            self._initialize_weights()
        
    def hierarchical_loss(self, id: torch.Tensor):
        """just returns 0, because temps are fitted freely"""
        return 0


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.individual_temps(x))
    
    def _initialize_weights(self) -> None:
        mn, std = -2.3, .5
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.normal_(mn, std)
    

class Scaling_ID_Random(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_participants: int,
        init_weights: bool = True,
    ):
        super(Scaling_ID, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.individual_temps = nn.Embedding(num_participants, 1)

        # Define learnable global mean & std
        self.global_mean = nn.Parameter(torch.ones(out_size))
        self.global_std = nn.Parameter(torch.ones(out_size))


        if init_weights:
            self._initialize_weights()
        
    def hierarchical_loss(self, id: torch.Tensor):
        """Encourage slopes to stay within a normal distribution."""
        return torch.mean((self.individual_temps(id) - self.global_mean) ** 2 / (2 * self.global_std**2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.individual_temps(x))
    
    def _initialize_weights(self) -> None:
        mn, std = -2.3, .5
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.normal_(mn, std)

class CombinedModel(nn.Module):
    def __init__(
            self,
            in_size: int,
            out_size: int,
            num_participants: int,
            scaling: str = "free",
            init_weights=True,
    ):
        super(CombinedModel, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.num_participants = num_participants
        self.init_weights = init_weights
        # embedding model with random by-participant dimension weights
        self.model1 = SPoSE_ID_Random(
            in_size=self.in_size,
            out_size=self.out_size, 
            num_participants=self.num_participants,
            init_weights=self.init_weights
        )
        if scaling == "free":
            # freely-varying by-participant softmax temperatures
            self.model2 = Scaling_ID(
                in_size=1, 
                out_size=1, 
                num_participants=num_participants,
                init_weights=self.init_weights
                )
        elif scaling == "random":
            # freely-varying by-participant softmax temperatures
            self.model2 = Scaling_ID_Random(
                in_size=1, 
                out_size=1, 
                num_participants=num_participants,
                init_weights=self.init_weights
                )

    def forward(self, x, id, distance_metric):
        x = self.model1(x, id)
        anchor, positive, negative = torch.unbind(torch.reshape(x, (-1, 3, self.out_size)), dim=1)
        sims_prep = ut.compute_similarities(anchor, positive, negative, "odd_one_out", distance_metric)
        sims = torch.stack(sims_prep, dim=-1)
        temp_scaling = self.model2(id[::3])
        sims_scaled = sims/temp_scaling
        loss = torch.mean(-torch.log(F.softmax(sims_scaled, dim=1)[:, 0]))
        return loss, anchor, positive, negative


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
