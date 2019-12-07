import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.neur.module import SVDLinear

class FNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, layer_count=4, layer_width=128, parametrization="svd", bias=True, batch_norm=True):
        super().__init__()
        self.parametrization = parametrization
        self.linearities = []

        self.first = self.initialize_linearity(input_dim, layer_width, bias)
        self.linearities.append(self.first)

        hidden = []
        for _layer in range(layer_count):
            linearity = self.initialize_linearity(layer_width, layer_width, bias)
            hidden.append(linearity)
            self.linearities.append(linearity)

            hidden.append(nn.ReLU())
            if batch_norm:
                hidden.append(nn.BatchNorm1d(layer_width))

        self.hidden = nn.Sequential(*hidden)

        self.last = self.initialize_linearity(layer_width, output_dim, bias)
        self.linearities.append(self.last)

    def initialize_linearity(self, input_dim, layer_width, bias):
        if self.parametrization == "svd":
            linearity = SVDLinear(input_dim, layer_width, bias)
        elif self.parametrization == "standard":
            linearity = nn.Linear(input_dim, layer_width, bias)
            nn.init.orthogonal_(linearity.weight)
        else:
            raise Exception("Unknown parametrization.  Must be 'svd' or 'standard'")

        return linearity

    def forward(self, inputs):
        out = self.first(inputs)
        out = F.relu(out)
        out = self.hidden(out)
        out = self.last(out)

        return out

    def singular_value_sets(self):
        if self.parametrization == "svd":
            return [linearity.Sweight for linearity in self.linearities]
        else:
            singular_value_sets = []
            for linearity in self.linearities:
                _, S, _ = torch.svd(linearity.weight, compute_uv=False)
                singular_value_sets.append(S)

            return singular_value_sets

    def weight_sets(self):
        if self.parametrization == "svd":
            raise Exception("Not implemented yet.")
        else:
            weight_sets = []
            for linearity in self.linearities:
                weight_sets.append(linearity.weight)

            return weight_sets
