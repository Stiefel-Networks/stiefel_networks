# stiefel_networks

Exploring the expressive power of neural networks with linearities constrained to the [Stiefel manifold](https://en.wikipedia.org/wiki/Stiefel_manifold)

# Definitions

- Strongly orthogonal weight matrix.  _Normalized_ W^T W = I
- Weakly orthogonal weight matrix - randomly initialized, and sufficiently large that _normalized_ W^T W ≈ I, no sabotage of any kind.
- Pathologically non-orthogonal weight matrix, where 
- Orthonormal weight matrix - an orthogonal weight matrix with unit length columns.  Note that no normalization is necessary for W^T W = I


- Strong orthogonal initialization
- Strong orthogonal training update (cayley transformation maybe exponentiation for curved sorta integration, Chiheb's work, Logan wants to do this)
- Weak orthogonal initialization (the convention)
- Weak orthogonal training update (by I - W^T W regularization)


# Hypotheses

- Non-orthogonal initialization increases time to convergence.
- Non-orthogonal initialization, preserved throughout training, reduces converged accuracy.
- A strictly orthogonal matrix of same size is of less capacity, reduces converged accuracy.
- Weak orthogonality is not maintained over unconstrained training.
- "Whole matrix, slight dendency" and "few columns nearly dependent in otherwise orthogonal matrix" are very different in terms of convergence rate and trained accuracy.
