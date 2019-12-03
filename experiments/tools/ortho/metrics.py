import numpy as np

from tools.ortho.matrix import normalize_columns


def cosine_similarity_matrix(matrix):
    """
    :param matrix:  Any square NxN matrix.
    :return:
        A matrix describing column-wise orthogonality of the input matrix.
        Specifically, element i,j is the cosine similarity of the columns
        i and j of the input matrix.
    """
    # We only care about the angles between the columns, not their magnitudes.
    # So normalize all of the columns.
    normalized_matrix = normalize_columns(matrix)

    # The inner product X^T X takes the dot product of each pair of columns.
    # Columns are normalized, so each element of X^T X is the cosine similarity
    # between two columns.
    inner_product = normalized_matrix.T @ normalized_matrix

    return inner_product


def total_cosine_simiarity(matrix):
    """
    :param matrix: Any square NxN matrix.
    :return:
        A scalar describing column-wise orthogonality of the input matrix.
        Specifically, return the total cosine similarity between the column
        vectors of the matrix.  A matrix where all columns are pairwise
        orthogonal has a total cosine similarity of 0.  A matrix where all
        columns exist along the same line (i.e. rank == 1) has a total
        cosine similarity of N * (N - 1) / 2, which is simply the number
        of elements above (or below) the main diagonal of the input.
    """
    cosine_similarities = cosine_similarity_matrix(matrix)

    # We are not interested in the diagonal entries (self-similarity is always
    # 1) and we do not want to double count the other pairs, so we zero all
    # elements on the main diagonal and below.
    upper_triangle = np.triu(cosine_similarities) - np.diag(np.diag(cosine_similarities))

    total_cosine_similarity = np.sum(upper_triangle)
    return total_cosine_similarity


def mean_cosine_similarity(matrix):
    """
    :param matrix: Any square NxN matrix
    :return:
        Same as total_cosine similarity, but normalized by the number of
        unique, non-self pairs of columns in the matrix: N * (N - 1) / 2
        An orthogonal matrix has mean cosine similarity of 0, and a matrix
        with rank 1 has mean cosine similarity of 1.

        For example, the following matrix

        [[ 1, 0, 0 ]
         [ 0, 2, 0 ]
         [ 0, 0, 3 ]]

        has mean cosine similarity 1.0, because all columns are orthogonal.
        Compare to the matrix

        [[ 1, 0, 0 ]
         [ 2, 0, 0 ]
         [ 3, 0, 0 ]]

        which has mean cosine similarity 0.0, because all columns are
        collinear.  Compare again to

        [[ 1, 0 ]
         [ 1, 1 ]]

        which has mean cosine similarity 0.7071.  There is only one cosine
        similarity in the matrix - the one between column 1 and column 2.
        They are 45 degrees apart, meaning cosine similarity of 0.7071.

    :WARNING:
        This currently explodes for any matrix with a column of zeros.
    """
    matrix_size = matrix.shape[0]
    number_of_non_self_pairs = matrix_size * (matrix_size - 1) / 2

    return total_cosine_simiarity(matrix) / number_of_non_self_pairs
