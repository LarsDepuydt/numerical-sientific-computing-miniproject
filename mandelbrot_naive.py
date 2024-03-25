import numpy as np

from util import is_stable, is_stable_datatype


dtype = np.float64

def naive_mandelbrot(C, num_iterations, T):
    M = np.zeros(C.shape, dtype=dtype)
    print("output", M.dtype)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            M[i, j] = is_stable(C[i, j], num_iterations, T)

    return M

def naive_mandelbrot_datatype(C_re, C_im, num_iterations, T, p_im, p_re):
    M = np.zeros([p_im, p_re], dtype=dtype)

    for i in range(p_im):
        for j in range(p_re):
            M[i, j] = is_stable_datatype(C_im[i], C_re[j], num_iterations, T)

    return M

