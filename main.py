import time
import numpy as np

from mandelbrot_dask import create_complex_grid, compute_mandelbrot_dask, test_dask_mandelbrot
from mandelbrot_multiprocessing import parallel_mandelbrot
from mandelbrot_naive import naive_mandelbrot
from mandelbrot_numpyVector import vectorized_mandelbrot
from util import complex_matrix, plot_mandelbrot

x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
p_im, p_re = 8000, 8000
num_iterations = 30
T = 2
dask_chunks = '70 MiB'
input_dtype = np.complex64

def time_execution_datatype(function, C_re, C_im):
    start_time = time.time()
    M = function(C_re, C_im, num_iterations, T, p_im, p_re)
    end_time = time.time()

    print(f"Function executed in {end_time - start_time} seconds.")
    return M


def time_execution(function, C):
    start_time = time.time()
    M = function(C, num_iterations, T)
    end_time = time.time()

    print(f"Function executed in {end_time - start_time} seconds.")
    return M


if __name__ == "__main__":
    C = complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re, input_dtype)
    print("input", C.dtype)

    for i in range(3):
        M1 = time_execution(naive_mandelbrot, C)
        plot_mandelbrot(M1)

        M3 = time_execution(vectorized_mandelbrot, C)

        M4, ex_time = parallel_mandelbrot(C, num_iterations, T)
        print(f"Function executed in {ex_time} seconds.")

        C_dask = create_complex_grid(C, dask_chunks)
        M5, ex_time2 = compute_mandelbrot_dask(C_dask, num_iterations, T)
        print(f"Function executed in {ex_time} seconds.")

    test_dask_mandelbrot(C, num_iterations, T)
