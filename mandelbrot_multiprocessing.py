import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

from util import is_stable

default_num_processes = 8
default_num_chunks = 8*6


def mandelbrot_chunk(args):
    C_chunk, num_iterations, T = args
    M_chunk = np.zeros(C_chunk.shape)

    for i in range(C_chunk.shape[0]):
        for j in range(C_chunk.shape[1]):
            M_chunk[i, j] = is_stable(C_chunk[i, j], num_iterations, T)

    return M_chunk


def split_into_horizontal_chunks(C, num_chunks):
    total_rows = C.shape[0]
    chunk_height = total_rows // num_chunks
    # Ensure all rows are covered by adding the remainder to the last chunk
    chunks = [C[i * chunk_height: (i + 1) * chunk_height, :] for i in range(num_chunks - 1)]
    # Add the last chunk, which includes any remaining rows
    chunks.append(C[(num_chunks - 1) * chunk_height: total_rows, :])

    return chunks


def parallel_mandelbrot(C, num_iterations, T, p=default_num_processes, n=default_num_chunks):
    C_chunks = split_into_horizontal_chunks(C, n)

    # Prepare arguments for each chunk
    args_list = [(chunk, num_iterations, T) for chunk in C_chunks]

    # Process each chunk in parallel
    with Pool(processes=p) as pool:
        M_chunks = pool.map(mandelbrot_chunk, args_list)

    # Reassemble the matrix from the chunks by stacking them vertically
    M = np.vstack(M_chunks)

    return M


def test_parallel_mandelbrot(C, num_iterations, T):
    num_processes_values = [1, 2, 4, 8, 16]
    chunk_multipliers = [1, 2, 4, 6, 8]  # Example multipliers for the number of chunks

    plt.figure(figsize=(10, 6))

    # Loop over each chunk multiplier
    for multiplier in chunk_multipliers:
        time_values = np.zeros(len(num_processes_values))

        # Loop over each number of processes
        for i, n in enumerate(num_processes_values):
            start_time = time.time()
            # Use the multiplier to determine the number of chunks
            parallel_mandelbrot(C, num_iterations, T, p=n, n=n * multiplier)
            end_time = time.time()

            time_values[i] = end_time - start_time

        # Plot the current set of time values
        plt.plot(num_processes_values, time_values, marker='o', linestyle='-', label=f'Chunks x{multiplier}')

    plt.xlabel('Number of Processes')
    plt.ylabel('Execution Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.xticks(num_processes_values)
    plt.show()
