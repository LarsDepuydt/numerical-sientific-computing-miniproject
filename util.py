import numpy as np
import matplotlib.pyplot as plt


def complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re):
    re = np.linspace(x_min, x_max, p_re)
    im = np.linspace(y_min, y_max, p_im)
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j

def complex_matrix_datatype(x_min, x_max, y_min, y_max, p_im, p_re, dtype):
    re = np.linspace(x_min, x_max, p_re, dtype=dtype)
    im = np.linspace(y_min, y_max, p_im, dtype=dtype)
    return re, im


def is_stable(c, num_iterations, T):
    z = 0.0

    for i in range(num_iterations):
        z = z ** 2 + c
        if abs(z) > T:
            return i / num_iterations

    return 1.0


def plot_mandelbrot(matrix):
    plt.imshow(matrix, cmap='hot', extent=[-2.0, 1.0, -1.5, 1.5])
    plt.show()


def is_stable_datatype(c_im, c_re, num_iterations, T):
    z_im = 0.0
    z_re = 0.0
    for i in range(num_iterations):
        z_re_sq = z_re**2
        z_im_sq = z_im**2
        if z_re_sq + z_im_sq > T**2:
            return i / num_iterations
        z_im = 2 * z_re * z_im + c_im
        z_re = z_re_sq - z_im_sq + c_re

    return 1.0
