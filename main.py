import time

from mandelbrot_naive import naive_mandelbrot
from mandelbrot_numba import numba_mandelbrot
from util import complex_matrix, plot_mandelbrot

x_min = -2.0
x_max = 1.0
y_min = -1.5
y_max = 1.5
p_im = 5000
p_re = 5000
num_iterations = 20
T = 2


def time_execution(function, C):
    start_time = time.time()
    M = function(C, x_min, x_max, y_min, y_max, p_im, p_re, num_iterations, T)
    end_time = time.time()

    print(f"Function executed in {end_time - start_time} seconds.")
    return M


C = complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re)

M1 = time_execution(naive_mandelbrot, C)
plot_mandelbrot(M1)
M2 = time_execution(numba_mandelbrot, C)
plot_mandelbrot(M2)
