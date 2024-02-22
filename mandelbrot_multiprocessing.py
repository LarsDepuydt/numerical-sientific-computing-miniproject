import numpy as np
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt

from util import is_stable

default_num_processes = 8
default_num_chunks = 1


def mandelbrot_chunk(C_chunk, num_iterations, T):
    M_chunk = np.zeros(C_chunk.shape)

    for i in range(C_chunk.shape[0]):
        for j in range(C_chunk.shape[1]):
            M_chunk[i, j] = is_stable(C_chunk[i, j], num_iterations, T)

    return M_chunk


def split_into_blocks(C, n):
    p_im, p_re = C.shape
    block_height = int(np.sqrt(p_im * p_re / n))
    block_width = block_height

    blocks = []
    positions = []
    for i in range(0, p_im, block_height):
        for j in range(0, p_re, block_width):
            block = C[i:min(i + block_height, p_im), j:min(j + block_width, p_re)]
            blocks.append(block)
            positions.append((i, j))  # Store the top-left position of the block
    return blocks, positions


def parallel_mandelbrot(C, num_iterations, T, p=default_num_processes, n=default_num_chunks):
    blocks, positions = split_into_blocks(C, n)
    args_for_pool = [(block, num_iterations, T) for block in blocks]

    with Pool(processes=p) as pool:
        results = pool.starmap(mandelbrot_chunk, args_for_pool)

    # Reassemble the results into a new matrix of the same shape as C
    M = np.zeros_like(C, dtype=float)
    for (block, (i, j)) in zip(results, positions):
        M[i:i + block.shape[0], j:j + block.shape[1]] = block

    return M


def test_parallel_mandelbrot(C, num_iterations, T):
    num_processes_values = [1, 2, 4, 8, 12, 16, 20, 30]
    chunk_multipliers = [1, 2, 4]  # Example multipliers for the number of chunks

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
