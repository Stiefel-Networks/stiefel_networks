import torch
import numpy as np
from math import pi
from ortho.metrics import mean_cosine_similarity


class MExp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        # First two terms X^0 + X^1 (0! = 1, 1! = 1)
        eX = torch.eye(X.shape[0]).to(X.device) + X

        # running power/factorial
        A = X

        # Rest of terms: + (X/2)*X + (X/3*X/2)*X + (X/4*X/3*X/2)*X ...
        for i in torch.arange(2, 100):
            A = torch.mm(A / i, X)
            eX = eX + A

        ctx.save_for_backward(eX)
        return eX

    @staticmethod
    def backward(ctx, grad_output):
        eX, = ctx.saved_tensors
        return grad_output.mm(eX)


mexp = MExp.apply  # convenient alias

def exp_map(B, x):
    X = torch.sum(B * x, 0)
    return mexp(X)

def ortho_basis(n):

    indices = torch.triu(torch.arange(n*n).reshape(n,n),diagonal=1)
    basis_indices = indices[indices>0]

    num = basis_indices.numel()

    B = torch.zeros(num, n*n)
    B[torch.arange(num), basis_indices] = 1
    B = B.reshape((num, n, n))
    print(B)
    return B - B.permute((0,2,1))

if __name__ == "__main__":
    print(ortho_basis(10))

    orth = exp_map(ortho_basis(10), torch.rand(45).reshape(-1,1,1))
    print(orth)

    print(np.linalg.norm(orth, axis=1))
    print(mean_cosine_similarity(orth.numpy()))
