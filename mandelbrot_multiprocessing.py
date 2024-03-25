import numpy as np
from multiprocessing import Pool
import time

default_num_processes = 8
default_num_chunks = 8*6


def mandelbrot_chunk(args):
    C_chunk, num_iterations, T = args

    Z = np.zeros(C_chunk.shape, dtype=np.complex64)
    M = np.full(C_chunk.shape, fill_value=num_iterations, dtype=np.float16)
    diverged = np.zeros(C_chunk.shape, dtype=bool)  # Keep track of elements that have diverged

    for i in range(num_iterations):
        # Perform the Mandelbrot iteration
        Z = np.where(~diverged, Z ** 2 + C_chunk, 0)

        # Update the diverged status
        new_diverged = np.abs(Z) > T
        diverged |= new_diverged  # Update elements that have diverged this iteration

        # Update M only for newly diverged elements
        M = np.where(new_diverged & (M == num_iterations), i, M)

    M /= num_iterations
    return M

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

    start_time = time.time()

    # Process each chunk in parallel
    with Pool(processes=p) as pool:
        M_chunks = pool.map(mandelbrot_chunk, args_list)

    # Reassemble the matrix from the chunks by stacking them vertically
    M = np.vstack(M_chunks)

    end_time = time.time()
    ex_time = end_time - start_time

    return M, ex_time