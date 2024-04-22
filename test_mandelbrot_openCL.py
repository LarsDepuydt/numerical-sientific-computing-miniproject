import numpy as np
import pytest
from mandelbrot_openCL import complex_matrix, plot_mandelbrot, mandelbrot_openCL


def test_complex_matrix_dimensions():
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.0, 1.0
    p_im, p_re = 10, 15
    matrix = complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re)
    assert matrix.shape == (p_im, p_re), "Matrix dimensions are incorrect."


def test_complex_matrix_type():
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.0, 1.0
    p_im, p_re = 10, 15
    matrix = complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re)
    assert np.iscomplexobj(matrix), "Matrix elements are not complex numbers."


def test_plot_mandelbrot_runs():
    # Create a simple 2x2 matrix for testing
    matrix = np.array([[0, 1], [1, 0]], dtype=np.float32)
    try:
        plot_mandelbrot(matrix)
    except Exception as e:
        pytest.fail(f"plot_mandelbrot raised an exception {e}")


def test_mandelbrot_openCL():
    C = complex_matrix(-2.0, 1.0, -1.0, 1.0, 10, 10)
    num_iterations = 100
    T = 2.0
    result, exec_time = mandelbrot_openCL(C, num_iterations, T)
    assert isinstance(result, np.ndarray), "Output is not an numpy array"
    assert result.shape == (10, 10), "Output shape is incorrect"
    assert isinstance(exec_time, float), "Execution time is not a float"
