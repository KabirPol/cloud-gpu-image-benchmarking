
import numpy as np
from numba import cuda
import math


# CPU implementation of Sobel edge detection
def sobel_cpu(image):
    img = image.astype(np.float32)

    height, width = img.shape

    gx = np.zeros((height, width), dtype=np.float32)
    gy = np.zeros((height, width), dtype=np.float32)

    gx[1:-1, 1:-1] = (
        -img[:-2, :-2] + img[:-2, 2:]
        - 2 * img[1:-1, :-2] + 2 * img[1:-1, 2:]
        - img[2:, :-2] + img[2:, 2:]
    )

    gy[1:-1, 1:-1] = (
        -img[:-2, :-2] - 2 * img[:-2, 1:-1] - img[:-2, 2:]
        + img[2:, :-2] + 2 * img[2:, 1:-1] + img[2:, 2:]
    )

    edges = np.sqrt(gx * gx + gy * gy)

    return edges


# CUDA kernel for Sobel edge detection
@cuda.jit
def sobel_kernel(img, edges):
    x, y = cuda.grid(2)

    height, width = img.shape

    if x > 0 and x < width - 1 and y > 0 and y < height - 1:

        gx = (
            -img[y-1, x-1] + img[y-1, x+1]
            - 2 * img[y, x-1] + 2 * img[y, x+1]
            - img[y+1, x-1] + img[y+1, x+1]
        )

        gy = (
            -img[y-1, x-1] - 2 * img[y-1, x] - img[y-1, x+1]
            + img[y+1, x-1] + 2 * img[y+1, x] + img[y+1, x+1]
        )

        edges[y, x] = math.sqrt(gx * gx + gy * gy)


# Function to execute Sobel edge detection on GPU
def sobel_gpu(image, threads_per_block=(16, 16)):

    if not cuda.is_available():
        raise RuntimeError("CUDA GPU is not available.")

    img = image.astype(np.float32)

    d_img = cuda.to_device(img)

    # Initialize output to zero so border pixels remain zero
    d_edges = cuda.to_device(np.zeros_like(img))

    blocks_per_grid = (
        (img.shape[1] + threads_per_block[0] - 1)
        // threads_per_block[0],

        (img.shape[0] + threads_per_block[1] - 1)
        // threads_per_block[1]
    )

    sobel_kernel[blocks_per_grid, threads_per_block](d_img, d_edges)

    cuda.synchronize()

    edges = d_edges.copy_to_host()

    return edges


# Test CPU and GPU implementations
def main():

    if not cuda.is_available():
        print("CUDA GPU is not available.")
        return

    print("GPU detected successfully!")
    print("GPU:", cuda.get_current_device().name)

    np.random.seed(42)

    image = np.random.randint(
        0, 256, size=(512, 512), dtype=np.uint8
    )

    cpu_edges = sobel_cpu(image)
    gpu_edges = sobel_gpu(image)

    print("\nImage shape:", image.shape)

    print(
        "Maximum absolute difference:",
        np.max(np.abs(cpu_edges - gpu_edges))
    )

    if np.allclose(cpu_edges, gpu_edges, atol=1e-5):
        print("Verification successful! CPU and GPU outputs match.")
    else:
        print("Verification failed!")


if __name__ == "__main__":
    main()
