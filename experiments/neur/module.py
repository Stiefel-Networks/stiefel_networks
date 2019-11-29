import torch
from torch.nn.parameter import Parameter
from math import pi
import torch.nn.init as init
import experiments.neur.functional as fn
import torch.nn.functional as F


class OrthoLinear(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, features, bias=True):
        super().__init__()
        self.basis = fn.ortho_basis(features)
        self.features = features

        self.weight = Parameter(torch.zeros(self.basis.shape[0]))
        if bias:
            self.bias = Parameter(torch.Tensor(features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.weight,-pi/2,pi/2)
        if self.bias is not None:
            #bound = 1 / math.sqrt(self.features)
            init.uniform_(self.bias, -0.1, 0.1)
            init.zeros_(self.bias)

    def forward(self, input):
        return fn.OrthoLinear(input,self.weight,self.bias)

class SVDLinear(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.Uweight = Parameter(torch.zeros(in_features*(in_features-1)//2))
        self.Vweight = Parameter(torch.zeros(out_features*(out_features-1)//2))
        self.Sweight = Parameter(torch.zeros(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.Uweight, -pi / 2, pi / 2)
        init.uniform_(self.Vweight, -pi / 2, pi / 2)
        init.ones_(self.Sweight)
        if self.bias is not None:
            #bound = 1 / math.sqrt(self.features)
            #init.uniform_(self.bias, -0.1, 0.1)
            init.zeros_(self.bias)

    def forward(self, input):
        return fn.SVDLinear(input,(self.in_features,self.out_features), self.Uweight, self.Sweight, self.Vweight,self.bias)

class SVDLinearWithInverse(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.Uweight = Parameter(torch.zeros(in_features*(in_features-1)//2))
        self.Vweight = Parameter(torch.zeros(out_features*(out_features-1)//2))
        self.Sweight = Parameter(torch.zeros(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.Uweight, -pi / 2, pi / 2)
        init.uniform_(self.Vweight, -pi / 2, pi / 2)
        init.constant_(self.Sweight,1)
        if self.bias is not None:
            #bound = 1 / math.sqrt(self.features)
            #init.uniform_(self.bias, -0.1, 0.1)
            init.zeros_(self.bias)

    def forward(self, input):
        out = input+ self.bias
        out = fn.SVDLinear(out, (self.in_features, self.out_features), self.Uweight, self.Sweight, self.Vweight,None)
        out = F.relu(out)

        out= fn.SVDLinear(out,(self.out_features,self.in_features), -self.Vweight, 1/self.Sweight, -self.Uweight,None)
        out -= self.bias
        return out

if __name__ == "__main__":
    print(ortho_basis(10))

    orth = exp_map(ortho_basis(10), torch.rand(45).reshape(-1,1,1))
    print(orth)

    print(orth.norm('nuc'))
