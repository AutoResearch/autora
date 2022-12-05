import numpy as np

from autora.experimentalist.sampler.dissimilarity import dissimilarity_sampler


def test_dissimilarity_sampler():
    # define two matrices
    matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    matrix2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    # reorder matrix1 according to its distances to matrix2
    reordered_matrix1 = dissimilarity_sampler(matrix1, matrix2, n=2)

    assert reordered_matrix1.shape[0] == 2
    assert reordered_matrix1.shape[1] == 3
    assert np.array_equal(reordered_matrix1, np.array([[10, 11, 12], [7, 8, 9]]))

    # print the reordered matrix
