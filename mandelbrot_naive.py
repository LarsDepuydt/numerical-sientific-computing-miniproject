import numpy as np

from util import is_stable


dtype = np.float64

def naive_mandelbrot(C, num_iterations, T):
    M = np.zeros(C.shape, dtype=dtype)
    print("output", M.dtype)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            M[i, j] = is_stable(C[i, j], num_iterations, T)

    return M
