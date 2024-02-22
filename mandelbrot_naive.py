import numpy as np

from util import complex_matrix, is_stable, plot_mandelbrot


def naive_mandelbrot(C, num_iterations, T):
    M = np.zeros(C.shape)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            M[i, j] = is_stable(C[i, j], num_iterations, T)

    return M
