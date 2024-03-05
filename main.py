import time
from multiprocessing import Pool

from mandelbrot_multiprocessing import parallel_mandelbrot, test_parallel_mandelbrot
from mandelbrot_naive import naive_mandelbrot
from mandelbrot_numba import numba_mandelbrot
from mandelbrot_numpyVector import vectorized_mandelbrot
from util import complex_matrix, plot_mandelbrot

x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
p_im, p_re = 5000, 5000
num_iterations = 30
T = 2


def time_execution(function, C):
    start_time = time.time()
    M = function(C, num_iterations, T)
    end_time = time.time()

    print(f"Function executed in {end_time - start_time} seconds.")
    return M


if __name__ == "__main__":
    C = complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re)

    M1 = time_execution(naive_mandelbrot, C)
    plot_mandelbrot(M1)

    M2 = time_execution(numba_mandelbrot, C)
    plot_mandelbrot(M2)

    M3 = time_execution(vectorized_mandelbrot, C)
    plot_mandelbrot(M3)

    M4 = time_execution(parallel_mandelbrot, C)
    plot_mandelbrot(M4)

    test_parallel_mandelbrot(C, num_iterations, T)
