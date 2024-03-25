import time
import numpy as np

from mandelbrot_dask import create_complex_grid, compute_mandelbrot_dask, test_dask_mandelbrot
from mandelbrot_naive import naive_mandelbrot, naive_mandelbrot_datatype
from util import complex_matrix, plot_mandelbrot, complex_matrix_datatype

x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
p_im, p_re = 5000, 5000
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
    # [C_re, C_im] = complex_matrix_datatype(x_min, x_max, y_min, y_max, p_im, p_re, dtype=dtype)

    for i in range(3):
        M1 = time_execution(naive_mandelbrot, C)

        # M2 = time_execution_datatype(naive_mandelbrot_datatype, C_re, C_im)
        # plot_mandelbrot(M1)

        # C_dask = create_complex_grid(C, dask_chunks)
        # M5, ex_time = compute_mandelbrot_dask(C_dask, num_iterations, T)
        # print(f"Function executed in {ex_time} seconds.")
        # plot_mandelbrot(M5)

    # test_dask_mandelbrot(C, num_iterations, T)
