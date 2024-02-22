from numba import jit
import numpy as np

from util import is_stable, plot_mandelbrot


@jit(nopython=True)
def complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re) -> np.ndarray:
    re = np.linspace(x_min, x_max, p_re)
    im = np.linspace(y_min, y_max, p_im)
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j


@jit(nopython=True)
def is_stable(c, num_iterations, T):
    z = 0
    l_c = 0
    for i in range(num_iterations):
        l_c = i
        z = z ** 2 + c
        if abs(z) > T:
            break

    return l_c / num_iterations


@jit(nopython=True)
def numba_mandelbrot(C, x_min, x_max, y_min, y_max, p_im, p_re, num_iterations, T):
    M = np.zeros((p_im, p_re))

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            M[i, j] = is_stable(C[i, j], num_iterations, T)

    return M
