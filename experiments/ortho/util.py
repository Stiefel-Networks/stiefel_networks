import numpy as np


def random_square_matrix(matrix_size):
    return np.random\
        .randn(matrix_size ** 2)\
        .reshape((matrix_size, matrix_size))


def append_random_row_and_column(matrix):
    new_matrix = random_square_matrix(matrix.shape[0] + 1)
    new_matrix[:-1, :-1] = matrix

    return new_matrix


def normalize_columns(matrix):
    matrix_column_norms = np.linalg.norm(matrix, axis=0)
    normalized = matrix / matrix_column_norms

    return normalized