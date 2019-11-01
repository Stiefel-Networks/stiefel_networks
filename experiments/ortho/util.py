import numpy as np


def random_square_matrix(matrix_size):
    return np.random\
        .randn(matrix_size ** 2)\
        .reshape((matrix_size, matrix_size))

def dependent_square_matrix(n,delta):
    vec = np.zeros(n*n)
    test = np.random.uniform(0,1,size=n*n)
    sampleInit = np.random.uniform(0,1,size=n*n)
    vec[0] = np.random.uniform(0,1)
    for i in range(1,n*n):
        if test[i] <= delta:
            vec[i] = vec[i-1]
        else:
            vec[i] = sampleInit[i]
    randomMat = vec.reshape((n,n))
    return randomMat

def append_random_row_and_column(matrix):
    new_matrix = random_square_matrix(matrix.shape[0] + 1)
    new_matrix[:-1, :-1] = matrix

    return new_matrix


def normalize_columns(matrix):
    matrix_column_norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / matrix_column_norms

    return normalized
