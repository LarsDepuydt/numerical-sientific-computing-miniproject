import numpy as np


def vectorized_mandelbrot(C, num_iterations, T):
    # Initialize Z (current value of iteration) and M (result matrix)
    Z = np.zeros(C.shape, dtype=np.complex128)
    M = np.full(C.shape, fill_value=num_iterations, dtype=np.float64)

    for i in range(num_iterations):
        # Mask to identify elements that haven't escaped
        not_escaped = np.abs(Z) <= T

        # Perform the iteration on elements that haven't escaped
        Z[not_escaped] = Z[not_escaped] ** 2 + C[not_escaped]

        # Update M for elements that have just escaped
        escape_mask = (np.abs(Z) > T) & (M == num_iterations)
        M[escape_mask] = i

    # Normalize the escape counts
    M /= num_iterations

    return M
