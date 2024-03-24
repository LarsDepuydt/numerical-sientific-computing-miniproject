import dask.array as da
from dask.distributed import Client
import numpy as np
import math
import time

from util import is_stable


def create_complex_grid(C, n_chunks):
    return da.from_array(C, chunks=n_chunks)


def mandelbrot_block(C_chunk, num_iterations, T):
    M_chunk = np.zeros(C_chunk.shape)

    for i in range(C_chunk.shape[0]):
        for j in range(C_chunk.shape[1]):
            M_chunk[i, j] = is_stable(C_chunk[i, j], num_iterations, T)

    return M_chunk


def dask_vectorized_mandelbrot(C_chunk, num_iterations, T):
    # Initialize Z (current value of iteration) and M (result matrix) as Dask arrays
    Z = da.zeros(C_chunk.shape, dtype=np.complex64)
    M = da.full(C_chunk.shape, fill_value=num_iterations, dtype=np.float32)

    for i in range(num_iterations):
        not_escaped = da.abs(Z) <= T
        Z = Z * not_escaped + (Z ** 2 + C_chunk) * not_escaped
        escape_mask = (da.abs(Z) > T) & (M == num_iterations)
        M = da.where(escape_mask, i, M)

    M /= num_iterations
    return M



def compute_mandelbrot_dask(C_da, num_iterations, T):
    client = Client()
    start_time = time.time()

    # Apply the Mandelbrot computation to each block
    M_da = C_da.map_blocks(dask_vectorized_mandelbrot, num_iterations, T, dtype=np.float16)

    M = M_da.compute()

    execution_time = time.time() - start_time
    client.close()

    return M, execution_time


# def test_parallel_mandelbrot(C, num_iterations, T):
#     num_processes_values = [1, 2, 4, 8, 16]
#     chunk_multipliers = [1, 2, 4, 6, 8]  # Example multipliers for the number of chunks
#
#     plt.figure(figsize=(10, 6))
#
#     # Loop over each chunk multiplier
#     for multiplier in chunk_multipliers:
#         time_values = np.zeros(len(num_processes_values))
#
#         # Loop over each number of processes
#         for i, n in enumerate(num_processes_values):
#             start_time = time.time()
#             # Use the multiplier to determine the number of chunks
#             parallel_mandelbrot(C, num_iterations, T, p=n, n=n * multiplier)
#             end_time = time.time()
#
#             time_values[i] = end_time - start_time
#
#         # Plot the current set of time values
#         plt.plot(num_processes_values, time_values, marker='o', linestyle='-', label=f'Chunks x{multiplier}')
#
#     plt.xlabel('Number of Processes')
#     plt.ylabel('Execution Time (seconds)')
#     plt.legend()
#     plt.grid(True)
#     plt.xticks(num_processes_values)
#     plt.show()
