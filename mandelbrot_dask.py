import dask.array as da
from dask.distributed import Client
import numpy as np
import matplotlib.pyplot as plt
import time

from util import is_stable, complex_matrix


def create_complex_grid(C, n_chunks):
    return da.from_array(C, chunks=n_chunks)


def mandelbrot_block(C_chunk, num_iterations, T):
    M_chunk = np.zeros(C_chunk.shape)

    for i in range(C_chunk.shape[0]):
        for j in range(C_chunk.shape[1]):
            M_chunk[i, j] = is_stable(C_chunk[i, j], num_iterations, T)

    return M_chunk


def dask_vectorized_mandelbrot(C_chunk, num_iterations, T):
    Z = da.zeros(C_chunk.shape, dtype=np.complex64)
    M = da.full(C_chunk.shape, fill_value=num_iterations, dtype=np.float16)
    diverged = da.zeros(C_chunk.shape, dtype=bool)  # Keep track of elements that have diverged

    for i in range(num_iterations):
        # Perform the Mandelbrot iteration
        Z = da.where(~diverged, Z ** 2 + C_chunk, 0)

        # Update the diverged status
        new_diverged = np.abs(Z) > T
        diverged |= new_diverged  # Update elements that have diverged this iteration

        # Update M only for newly diverged elements
        M = da.where(new_diverged & (M == num_iterations), i, M)

    M /= num_iterations
    return M


def compute_mandelbrot_dask(C_da, num_iterations, T):
    client = Client('130.225.37.203:8786')
    start_time = time.time()

    # Apply the Mandelbrot computation to each block
    M_da = C_da.map_blocks(dask_vectorized_mandelbrot, num_iterations, T, dtype=np.float16)

    M = M_da.compute()

    execution_time = time.time() - start_time
    client.close()

    return M, execution_time


def test_dask_mandelbrot(C, num_iterations, T):
    # Range of chunk sizes to test
    chunk_sizes = np.arange(10, 101, 10)  # 10 MiB to 100 MiB, inclusive
    execution_times = []

    for size in chunk_sizes:
        times_for_size = []
        for _ in range(3):  # Run each test 3 times
            # Convert size to string format with MiB
            dask_chunks = f"{size} MiB"

            # Your grid creation might vary; ensure it accepts dask_chunks correctly
            C_dask = create_complex_grid(C, dask_chunks)  # Adjust as per your function's definition

            # Execute the Mandelbrot computation
            _, ex_time = compute_mandelbrot_dask(C_dask, num_iterations, T)

            # Collect execution time
            times_for_size.append(ex_time)

        # Store the average execution time for this chunk size
        execution_times.append(np.mean(times_for_size))

    # Plotting the results
    print(chunk_sizes)
    print(execution_times)


def plot_result(input, result):
    plt.figure(figsize=(10, 6))
    plt.plot(input, result, marker='o', linestyle='-')
    plt.xlabel('Chunk Size (MiB)')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Mandelbrot Computation Time vs. Dask Chunk Size')
    plt.grid(True)
    plt.show()

x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
p_im, p_re = 8000, 8000
num_iterations = 30
T = 2
dask_chunks = '70 MiB'

if __name__ == "__main__":
    # C = complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re)
    # test_dask_mandelbrot(C, num_iterations, T)

    chunk_sizes = np.arange(10, 201, 10)
    result = [13.500008424123129, 11.866398731867472, 11.847947438557943, 11.34710399309794, 11.33056608835856, 11.385741472244263, 10.199405590693155, 10.600943247477213, 10.701882441838583, 10.49396808942159, 10.372452735900879, 10.425958315531412, 10.44973889986674, 10.363054434458414, 10.212701400121054, 10.164047082265219, 10.216555118560791, 10.230491638183594, 10.227827548980713, 10.4537992477417]
    plot_result(chunk_sizes, result)
