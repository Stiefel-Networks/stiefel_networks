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
    """
    Does the same thing that ortho basis does, but less memory.
    Make a mesh grid, which is just a matrix whose elements express their positions
    allocate nxn zeros.
    Wherever x < y, slam x in.
    :param n: Size of of orthogonal matrix.
    :param x:
        Minimal parametrization of orthogonal matrix.
        Should be size n * (n - 1) / 2.
        Can be any values, but nice values are in [-pi, pi]
    :return: Orthogonal matrix.
    """
    mx, my = torch.meshgrid(torch.arange(n),torch.arange(n))
    X = torch.zeros((n,n),dtype=torch.float32,device=x.device)
    X[mx<my] = x
    X = X-X.permute(1,0)
    return expm_skew(X)

def OrthoLinear(input, weight, bias):
    basis = ortho_basis(input.shape[len(input.shape) - 1])
    W = exp_map(basis, weight.reshape(-1, 1, 1))
    return F.linear(input, W, bias)

def SVDLinear(input, dim, Uweight, Sweight, Vweight, bias):
    """
    This (right now) needs matrix with dim[1] < dim[0] (i.e. W is wide).  If it is not wide,
    then
    :param input:
        NxD, N is batch size, D is size of vector.
    :param dim:
        Dimensions of weight matrix before SVD.
    :param Uweight:
        dim[0] * (dim[0] - 1) / 2 vector of weights, as per exp_map
    :param Sweight:
        dim[1]
    :param Vweight:
        Same as Uweight, but with dim[1].
    :param bias:
    :return:
    """
    # If wide (less rows than columns), then construct W transpose, rather than W
    wide_W = dim[0] < dim[1]
    if wide_W:
        dim = dim[::-1]
        tmp = Uweight
        Uweight = Vweight
        Vweight = tmp

    # Chop of the zero'd off columns of U
    U = exp_map(dim[0], Uweight)[:,:Sweight.numel()]
    V = exp_map(dim[1], Vweight)

    # Use Hadamard because it's the same for diagonals if they're the right shape.
    W = torch.mm(U*Sweight.reshape(1,-1), V.t())

    if wide_W:
        W = W.t()

    if bias is not None:
        return torch.addmm(bias, input, W)
    else:
        return torch.mm(input, W)


if __name__ == "__main__":
    from math import pi

    # This is all built on the convention of input and activation being row vectors, and weights being right multiplied.
    # So number of columns in W is the output dimension.  Number of rows is input dimension.

    # For U and V, that means U is the first matrix to touch x, and should be input_dim x input_dim. (V is output_dim x output_dim).

    # W is 4x3, so x * W is in R**3  (Recall: Row vectors!!)
    skinny_reparametrized = SVDLinear(
        torch.tensor([1., 2, 3, 4]).reshape(1, 4),
        (4, 3),
        torch.tensor([0., 0, 0, 0, 0, 0]), # Recall: n(n-1)/2.  Also, e ** 0 is 1, i.e. this is makes the identity matrix.  weird.
        torch.tensor([1., 1, 1]),
        torch.tensor([pi / 2., 0, 0]),
        torch.tensor([0.5, 0.5, 0.5]).reshape(1, 3),
    )
    print(skinny_reparametrized)

    wide_reparametrized = SVDLinear(
        torch.tensor([1., 2, 3]).reshape(1, 3),
        (3, 4),
        torch.tensor([pi / 2., 0, 0]),
        torch.tensor([1., 1, 1]),
        torch.tensor([0., 0, 0, 0, 0, 0]), # Recall: n(n-1)/2.  Also, e ** 0 is 1, i.e. this is makes the identity matrix.  weird.
        None,
    )
    print(wide_reparametrized)
