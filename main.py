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


def naive_mandelbrot():
    p_re = 5000
    p_im = 5000
    C = complex_matrix(-2.0, 1.0, -1.5, 1.5, p_im, p_re)
    M = np.zeros((p_im, p_re))

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            M[i, j] = is_stable(C[i, j], num_iterations=20, T=2)

    plot_mandelbrot(M)


naive_mandelbrot()
