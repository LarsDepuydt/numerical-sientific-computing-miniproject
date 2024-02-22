from numba import jit
import numpy as np


@jit(nopython=True)
def is_stable_numba(c, num_iterations, T):
    z = 0
    l_c = 0
    for i in range(num_iterations):
        l_c = i
        z = z ** 2 + c
        if abs(z) > T:
            break

    return l_c / num_iterations


@jit(nopython=True)
def numba_mandelbrot(C, num_iterations, T):
    M = np.zeros(C.shape)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            M[i, j] = is_stable_numba(C[i, j], num_iterations, T)

    return M
