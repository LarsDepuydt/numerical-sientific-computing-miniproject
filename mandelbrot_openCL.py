"""
This is the "mandelbrot_openCL" module.

The module provides a set of functions to compute and visualize the Mandelbrot set using OpenCL for accelerated computation.
It includes functions to create a complex number grid, perform the computation using OpenCL, visualize the results, and benchmark performance across different computational configurations.
"""

import numpy as np
import pyopencl as cl
import time
import matplotlib.pyplot as plt


def complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re):
    """
    Generates a 2D grid of complex numbers across a specified range on the complex plane.

    Parameters
    ----------
    x_min : float
        The minimum value on the real axis.
    x_max : float
        The maximum value on the real axis.
    y_min : float
        The minimum value on the imaginary axis.
    y_max : float
        The maximum value on the imaginary axis.
    p_im : int
        The number of points along the imaginary axis.
    p_re : int
        The number of points along the real axis.

    Returns
    -------
    np.ndarray
        A 2D numpy array of complex numbers, representing points on the complex plane.
    """
    re = np.linspace(x_min, x_max, p_re, dtype=np.float64)
    im = np.linspace(y_min, y_max, p_im, dtype=np.float64)
    return (re[np.newaxis, :] + im[:, np.newaxis] * 1j).astype(np.complex64)


def plot_mandelbrot(matrix):
    """
    Displays a visualization of the Mandelbrot set using a heat map.

    Parameters
    ----------
    matrix : np.ndarray
        A 2D numpy array containing the escape times or iterations to boundary for each point in the complex plane matrix.

    Returns
    -------
    None
    """
    plt.imshow(matrix, cmap='hot', extent=[-2.0, 1.0, -1.5, 1.5])
    plt.show()


def mandelbrot_openCL(C, num_iterations, T, local_size_input=None):
    """
       Computes the Mandelbrot set using an OpenCL kernel on a grid of complex numbers.

       Parameters
       ----------
       C : np.ndarray
           A 2D array of complex numbers representing points in the complex plane.
       num_iterations : int
           The maximum number of iterations for determining the stability of each point.
       T : float
           The escape threshold to determine if a point tends to infinity.
       local_size_input : tuple, optional
           The size of work groups for OpenCL computation; defaults to None, which lets OpenCL decide.

       Returns
       -------
       tuple
           The resulting numpy array of the Mandelbrot set calculation (reshaped to the original dimensions of C) and the computation's execution time.
       """
    p_im, p_re = C.shape
    M = np.empty(p_im * p_re, dtype=np.float32)  # Flattened array for the result

    # OpenCL setup
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    # Convert complex matrix to float array
    C_real_imag = np.column_stack((C.real.flatten(), C.imag.flatten())).astype(np.float32)

    kernel_code = """
    __kernel void mandelbrot(__global const float2 *C, __global float *M, const int max_iters, const float threshold) {
        int gid = get_global_id(0);
        float2 c = C[gid];
        float2 z = (float2)(0, 0);
        int n = 0;
        for(n = 0; n < max_iters; n++) {
            float x = z.x*z.x - z.y*z.y + c.x;
            float y = 2*z.x*z.y + c.y;
            if(x*x + y*y > threshold*threshold)
                break;
            z.x = x;
            z.y = y;
        }
        M[gid] = (float)n / (float)max_iters;
    }
    """
    prg = cl.Program(ctx, kernel_code).build()

    # Create buffers
    mf = cl.mem_flags
    C_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=C_real_imag)
    M_buf = cl.Buffer(ctx, mf.WRITE_ONLY, M.nbytes)
    global_size = (C_real_imag.shape[0],)  # Total number of elements in the flattened array

    # Execute the kernel
    start_time = time.time()
    prg.mandelbrot(queue, global_size, local_size_input, C_buf, M_buf, np.int32(num_iterations), np.float32(T))
    cl.enqueue_copy(queue, M, M_buf).wait()
    end_time = time.time()

    # Reshape the result to the original grid shape
    M = M.reshape(p_im, p_re)

    # Output results
    execution_time = end_time - start_time

    return M, execution_time


def benchmark_mandelbrot(C, num_iterations, T):
    """
    Benchmarks the Mandelbrot set computation across different local sizes of OpenCL work groups.

    Parameters
    ----------
    C : np.ndarray
        The complex plane array on which the Mandelbrot set is computed.
    num_iterations : int
        The maximum number of iterations per point.
    T : float
        The escape threshold used to determine if a point belongs to the Mandelbrot set.

    Returns
    -------
    None
    """
    local_sizes = [None, (32,), (64,), (128,), (256,)]
    execution_times = []

    for local_size in local_sizes:
        times = []
        for _ in range(3):  # Run the function 3 times for each local_size
            _, exec_time = mandelbrot_openCL(C, num_iterations, T, local_size)
            times.append(exec_time)
        average_time = np.mean(times)
        execution_times.append(average_time)
        print(f"Local size {local_size}: {average_time} seconds")

    labels = [str(ls[0]) if ls is not None else 'None' for ls in local_sizes]
    sizes = [ls[0] if ls is not None else 0 for ls in local_sizes]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(sizes, execution_times, marker='o', linestyle='-', color='blue') # Convert local sizes to scalar values for plotting
    plt.xticks(sizes, labels)  # Set custom ticks based on the local sizes
    plt.xlabel('Local Size')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Performance Benchmark of Mandelbrot Set Computation')
    plt.show()


if __name__ == "__main__":
    x_min, x_max, y_min, y_max = -2.0, 1.0, -1.5, 1.5
    p_im, p_re = 5000, 5000
    num_iterations = 30
    T = 2.0
    local_size = (64,)

    C = complex_matrix(x_min, x_max, y_min, y_max, p_im, p_re)
    print("input type", C.dtype)

    # for i in range(3):
    #     M6, execution_time = mandelbrot_openCL
    #     print(f"Computed the Mandelbrot set in {execution_time} seconds.")
    #     # plot_mandelbrot(M6)

    benchmark_mandelbrot(C, num_iterations, T)
