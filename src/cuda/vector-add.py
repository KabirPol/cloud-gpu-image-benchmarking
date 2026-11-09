
import numpy as np
from numba import cuda


# CUDA kernel for vector addition
@cuda.jit
def vector_add(A, B, C):
    i = cuda.grid(1)

    if i < C.size:
        C[i] = A[i] + B[i]


def main():

    # Check CUDA availability
    if not cuda.is_available():
        print("CUDA GPU is not available.")
        return

    print("GPU detected successfully!")
    print("GPU:", cuda.get_current_device().name)

    # Create input arrays in CPU memory
    A = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    B = np.array([10, 20, 30, 40, 50], dtype=np.float32)

    C = np.zeros_like(A)

    print("\nInput Array A:", A)
    print("Input Array B:", B)

    # Transfer arrays to GPU memory
    d_A = cuda.to_device(A)
    d_B = cuda.to_device(B)
    d_C = cuda.to_device(C)

    # Configure threads and blocks
    threads_per_block = 128

    blocks_per_grid = (
        A.size + threads_per_block - 1
    ) // threads_per_block

    print("\nThreads per block:", threads_per_block)
    print("Blocks per grid:", blocks_per_grid)

    # Execute CUDA kernel
    vector_add[blocks_per_grid, threads_per_block](
        d_A, d_B, d_C
    )

    # Copy result back to CPU memory
    C = d_C.copy_to_host()

    # Display results
    print("\nOutput Array C:", C)

    # Verify the result
    expected = A + B

    if np.array_equal(C, expected):
        print("Verification successful!")
    else:
        print("Verification failed!")


if __name__ == "__main__":
    main()
