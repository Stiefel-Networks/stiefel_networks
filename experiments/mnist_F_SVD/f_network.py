import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.neur.module import SVDLinear

class FNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, layer_count=4, layer_width=128, parametrization="svd", bias=True, batch_norm=True):
        super().__init__()
        self.parametrization = parametrization
        self.linearities = []

        Linearity = None
        if parametrization == "svd":
            Linearity = SVDLinear
        elif parametrization == "standard":
            Linearity = nn.Linear
            # TODO we should remove an experimental variable by orthogonally initializing all of these
        else:
            raise Exception("Unknown parametrization.  Must be 'svd' or 'standard'")

        self.first = Linearity(input_dim, layer_width, bias)
        self.linearities.append(self.first)

        hidden = []
        for _layer in range(layer_count):
            linearity = Linearity(layer_width, layer_width, bias)
            hidden.append(linearity)
            self.linearities.append(linearity)

            hidden.append(nn.ReLU())
            # TODO: are we sure we should be batchnorming?
            if batch_norm:
                hidden.append(nn.BatchNorm1d(layer_width))

        self.hidden = nn.Sequential(*hidden)

        self.last = Linearity(layer_width, output_dim, bias)
        self.linearities.append(self.last)

    def forward(self, inputs):
        out = self.first(inputs)
        out = F.relu(out)
        out = self.hidden(out)
        out = self.last(out)

        # TODO: this might be the right place to construct a softmax, but not right now!
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
