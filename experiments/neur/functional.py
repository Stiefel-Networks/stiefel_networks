import torch
from neur.expm import expm_skew
import torch.nn.functional as F

def ortho_basis(n):

    indices = torch.triu(torch.arange(n*n).reshape(n,n),diagonal=1)
    basis_indices = indices[indices>0]

    num = basis_indices.numel()

    B = torch.zeros(num, n*n)
    B[torch.arange(num), basis_indices] = 1
    B = B.reshape((num, n, n))
    return B - B.permute((0,2,1))

def exp_map(n, x):
    mx, my = torch.meshgrid(torch.arange(n),torch.arange(n))
    X = torch.zeros((n,n),dtype=torch.float32,device=x.device)
    X[mx<my]=x
    X = X-X.permute(1,0)
    return expm_skew(X)

def OrthoLinear(input, weight, bias):
    basis = ortho_basis(input.shape[len(input.shape) - 1])
    W = exp_map(basis, weight.reshape(-1, 1, 1))
    return F.linear(input, W, bias)

def SVDLinear(input, dim, Uweight, Sweight, Vweight, bias):
    U = exp_map(dim[0], Uweight)[:,:Sweight.numel()]
    V = exp_map(dim[1], Vweight)
    W = torch.mm(U*Sweight.reshape(1,-1), V.t())
    if bias is not None:
        return torch.addmm(bias, input, W)
    else:
        return torch.mm(input,W.t())



