import torch


class MExp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X):
        # First two terms X^0 + X^1 (0! = 1, 1! = 1)
        eX = torch.eye(X.shape[0]).to(X.device) + X

        # running power/factorial
        A = X

        # Rest of terms: + (X/2)*X + (X/3*X/2)*X + (X/4*X/3*X/2)*X ...
        for i in torch.arange(2, 40):
            A = torch.mm(A / i, X)
            eX = eX + A

        ctx.save_for_backward(eX)
        return eX

    @staticmethod
    def backward(ctx, grad_output):
        eX, = ctx.saved_tensors
        return grad_output.mm(eX)


mexp = MExp.apply  # convenient alias


if __name__ == "__main__":
    import numpy as np

    from tools.neur.functional import ortho_basis, exp_map
    from tools.ortho.metrics import mean_cosine_similarity

    print(ortho_basis(10))

    # TODO Logan this call is broken but I left it here in case you think that indicates a problem.
    orth = exp_map(ortho_basis(10), torch.rand(45).reshape(-1,1,1))
    print(orth)

    print(np.linalg.norm(orth, axis=1))
    print(mean_cosine_similarity(orth.numpy()))
