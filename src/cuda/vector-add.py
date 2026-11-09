
import numpy as np
from numba import cuda

# CUDA kernel
@cuda.jit
def vector_add(A, B, C):
    i = cuda.grid(1)

    if i < C.size:
        C[i] = A[i] + B[i]


def main():

    # Input arrays
    A = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    B = np.array([10, 20, 30, 40, 50], dtype=np.float32)

    C = np.zeros_like(A)

    # Transfer arrays to GPU
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)

    # Define execution configuration
    threads_per_block = 128
    blocks_per_grid = (A.size + threads_per_block - 1) // threads_per_block

    # Launch GPU kernel
    vector_add[blocks_per_grid, threads_per_block](d_A, d_B, d_C)

    # Copy result back to CPU
    C = d_C.copy_to_host()

    print("Array A:", A)
    print("Array B:", B)
    print("Result C:", C)


if __name__ == "__main__":
    main()
