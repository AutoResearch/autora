import numpy as np

from autora.experimentalist.sampler.inequality import summed_inequality_sampler


def test_inequality_sampler_1D():

    num_samples = 2

    # define two matrices
    matrix1 = np.array([1, 2, 3, 4, 5])
    matrix2 = np.array([1, 2, 3])

    # reorder matrix1 according to its distances to matrix2
    reordered_matrix1 = summed_inequality_sampler(matrix1, matrix2, n=num_samples)

    assert reordered_matrix1.shape[0] == num_samples
    assert reordered_matrix1.shape[1] == 1
    assert np.array_equal(reordered_matrix1, np.array([[4], [5]]))


def test_inequality_sampler_ND():
    # define two matrices
    matrix1 = np.array([[1, 1, 1], [2, 2, 2], [7, 8, 9], [10, 11, 12]])
    matrix2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    # reorder matrix1 according to its distances to matrix2
    reordered_matrix1 = summed_inequality_sampler(matrix1, matrix2, n=2)

    assert reordered_matrix1.shape[0] == 2
    assert reordered_matrix1.shape[1] == 3
    assert np.array_equal(reordered_matrix1, np.array([[7, 8, 9], [10, 11, 12]]))



