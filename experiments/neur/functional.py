import torch
from experiments.neur.expm import expm_skew
import torch.nn.functional as F

def ortho_basis(n):

    indices = torch.triu(torch.arange(n*n).reshape(n,n),diagonal=1)
    basis_indices = indices[indices>0]

    num = basis_indices.numel()

    B = torch.zeros(num, n*n)
    B[torch.arange(num), basis_indices] = 1
    B = B.reshape((num, n, n))
    return B - B.permute((0,2,1))

def exp_map(B, x):
    X = torch.sum(B * x, 0)
    return expm_skew(X)

def OrthoLinear(input, weight, bias):
    basis = ortho_basis(input.shape[len(input.shape) - 1])
    W = exp_map(basis, weight.reshape(-1, 1, 1))
    return F.linear(input, W, bias)

def SVDLinear(input, dim, Uweight, Sweight, Vweight, bias):
    Ubasis = ortho_basis(dim[0])
    Vbasis = ortho_basis(dim[1])
    U = exp_map(Ubasis, Uweight)
    V = exp_map(Vbasis, Vweight)

    W = torch.mm(U*Sweight.reshape(1,-1), V.t())
    if bias is not None:
        return torch.addmm(bias, input, W.t())
    else:
        return torch.mm(input,W.t())



