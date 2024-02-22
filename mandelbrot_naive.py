import numpy as np

from util import complex_matrix, is_stable, plot_mandelbrot


def naive_mandelbrot(C, x_min, x_max, y_min, y_max, p_im, p_re, num_iterations, T):
    M = np.zeros((p_im, p_re))

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            M[i, j] = is_stable(C[i, j], num_iterations, T)

    return M
