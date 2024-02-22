import numpy as np
import matplotlib.pyplot as plt


def complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re):
    re = np.linspace(x_min, x_max, p_re)
    im = np.linspace(y_min, y_max, p_im)
    return re[np.newaxis, :] + im[:, np.newaxis] * 1j


def is_stable(c, num_iterations, T):
    z = 0
    l_c = 0
    for i in range(num_iterations):
        l_c = i
        z = z ** 2 + c
        if abs(z) > T:
            break

    return l_c / num_iterations


def plot_mandelbrot(matrix):
    plt.imshow(matrix, cmap='hot', extent=[-2.0, 1.0, -1.5, 1.5])
    plt.show()
