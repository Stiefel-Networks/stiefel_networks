import torch
from torch.nn.parameter import Parameter
from math import pi
import torch.nn.init as init
import tools.neur.functional as fn
import torch.nn.functional as F


class OrthoLinear(torch.nn.Module):
    """
    This is in rough shape right now.  Tread with caution.
    """
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
        self.Sweight = Parameter(torch.zeros(min(in_features, out_features)))
        if bias:
            self.bias = Parameter(torch.zeros((1, out_features)))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # It is sufficient to only initialize across rotations on the half sphere (pi / 2) because we allow
        # negative singular values, which reflect.  There is a benefit, because the exponential map works better the
        # closer we are to the origin.
        init.uniform_(self.Uweight, -pi / 2, pi / 2)
        init.uniform_(self.Vweight, -pi / 2, pi / 2)

        # We initialize singular values to ones because we like orthogonal initializations, but these could be the
        # singular values corresponding to any old matrix if desired.
        init.ones_(self.Sweight)
        if self.bias is not None:
            # We don't initialize with a bias - it reduces parameter coupling on the first pass.  Let the network figure it out.
            init.zeros_(self.bias)

    def forward(self, input):
        return fn.SVDLinear(
            input,
            (self.in_features, self.out_features),
            self.Uweight,
            self.Sweight,
            self.Vweight,
            self.bias,
        )

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

    # Test callabilitiy

    svd_module = SVDLinear(3, 4)

    inputs = torch.tensor([
        [1., 2, 3],
        [4, 5, 6],
    ])
    inputs = inputs / torch.norm(inputs, dim=1, keepdim=True)
    outputs = svd_module(inputs)

    print(outputs)

    # Orthogonal initialization should preserve norms.
    print(torch.norm(outputs, dim=1))


    # Test optimizability

    # We set our target to be orthogonal, because if the exponential map can find an orthogonal target, then it can
    # trivially find any other.  i.e. this is the "difficult" case.
    target_mat = torch.zeros((3, 3))
    torch.nn.init.orthogonal_(target_mat)

    source_mat = torch.zeros_like(target_mat)
    torch.nn.init.orthogonal_(source_mat)

    svd_module = SVDLinear(3, 3)
    optimizer = torch.optim.SGD(svd_module.parameters(), lr=0.1)

    for i in range(10000):
        optimizer.zero_grad()

        estimate_mat = svd_module(source_mat)
        loss = torch.nn.functional.mse_loss(estimate_mat, target_mat)

        if i % 100 == 0:
            print(loss)

        loss.backward()

        optimizer.step()

    print(torch.norm(svd_module(source_mat), dim=0))
    print(torch.norm(svd_module(source_mat), dim=1))