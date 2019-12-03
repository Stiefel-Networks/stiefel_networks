import numpy as np


def random_square_matrix(matrix_size):
    return np.random\
        .randn(matrix_size ** 2)\
        .reshape((matrix_size, matrix_size))

#Created By Alex, this function creates a dependent random matrix
def dependent_square_matrix(n,delta):#takes in n(size of square matrix) and delta(percent of elements in matrix that are dependent
    vec = np.zeros(n*n)#initializes our random dependent matrix
    test = np.random.uniform(0,1,size=n*n)#stores our test values for deciding which elements should be dependent
    sampleInit = np.random.uniform(0,1,size=n*n)#stores our random matrix values
    vec[0] = np.random.uniform(0,1)#initializes the first element of our random matrix
    for i in range(1,n*n):
        if test[i] <= delta: #if our random test value is less than delta then
            vec[i] = vec[i-1] #make the current element of the random matrix the previous element
        else: 
            vec[i] = sampleInit[i] #set the current element of the random matrix to a independent random number
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
