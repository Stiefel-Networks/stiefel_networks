import torch.nn as nn
import torch.nn.functional as F

from tools.neur.module import SVDLinear

class FSVD(nn.Module):

    def __init__(self, input_dim, output_dim, layer_count=4, layer_width=128, bias=True, batch_norm=True):
        super().__init__()
        self.linearities = []

        self.first = SVDLinear(input_dim, layer_width)
        self.linearities.append(self.first)

        hidden = []
        for _layer in range(layer_count):
            svd_linearity = SVDLinear(layer_width, layer_width, bias)
            hidden.append(svd_linearity)
            self.linearities.append(svd_linearity)

            hidden.append(nn.ReLU())
            # TODO: are we sure we should be batchnorming?
            if batch_norm:
                hidden.append(nn.BatchNorm1d(layer_width))

        self.hidden = nn.Sequential(*hidden)

        self.last = SVDLinear(layer_width, output_dim)
        self.linearities.append(self.last)

    def forward(self, inputs):
        out = self.first(inputs)
        out = F.relu(out)
        out = self.hidden(out)
        out = self.last(out)

        # TODO: this might be the right place to construct a softmax, but not right now!
        return out

    def singular_value_sets(self):
        return [linearity.Sweight for linearity in self.linearities]
