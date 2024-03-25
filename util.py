import numpy as np
import matplotlib.pyplot as plt


def complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re, dtype):
    re = np.linspace(x_min, x_max, p_re, dtype=np.float64)
    im = np.linspace(y_min, y_max, p_im, dtype=np.float64)
    return (re[np.newaxis, :] + im[:, np.newaxis] * 1j).astype(dtype)


def is_stable(c, num_iterations, T):
    z = 0.0

    for i in range(num_iterations):
        z = z ** 2 + c

        # We return early if we hit the threshold and not waste any time
        if abs(z) > T:
            return i / num_iterations

    return 1.0


def plot_mandelbrot(matrix):
    plt.imshow(matrix, cmap='hot', extent=[-2.0, 1.0, -1.5, 1.5])
    plt.show()
