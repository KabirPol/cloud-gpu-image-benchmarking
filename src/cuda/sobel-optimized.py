
import numpy as np
from numba import cuda, float32
import math


# --------------------------------------------------
# CPU implementation: Sobel edge detection
# --------------------------------------------------

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

    return np.sqrt(gx * gx + gy * gy)


# --------------------------------------------------
# Baseline GPU kernel: Global memory
# --------------------------------------------------

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


# --------------------------------------------------
# Optimized GPU kernel: Shared-memory tiling
# Fixed block size: 16 x 16
# Shared tile size: 18 x 18 (includes halo)
# --------------------------------------------------

@cuda.jit
def sobel_shared_kernel(img, edges):

    # Thread coordinates within the block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    # Global coordinates of the first pixel in the block
    base_x = cuda.blockIdx.x * 16
    base_y = cuda.blockIdx.y * 16

    height, width = img.shape

    # Shared memory tile including one-pixel halo
    tile = cuda.shared.array(shape=(18, 18), dtype=float32)

    # Cooperatively load the tile and halo pixels
    for sy in range(ty, 18, 16):
        for sx in range(tx, 18, 16):

            gx = base_x + sx - 1
            gy = base_y + sy - 1

            if gx >= 0 and gx < width and gy >= 0 and gy < height:
                tile[sy, sx] = img[gy, gx]
            else:
                tile[sy, sx] = 0.0

    # Ensure all threads finish loading before computation
    cuda.syncthreads()

    # Global coordinates of the output pixel
    x = base_x + tx
    y = base_y + ty

    if x > 0 and x < width - 1 and y > 0 and y < height - 1:

        # Center position in shared memory
        sx = tx + 1
        sy = ty + 1

        gx = (
            -tile[sy-1, sx-1] + tile[sy-1, sx+1]
            - 2 * tile[sy, sx-1] + 2 * tile[sy, sx+1]
            - tile[sy+1, sx-1] + tile[sy+1, sx+1]
        )

        gy = (
            -tile[sy-1, sx-1] - 2 * tile[sy-1, sx] - tile[sy-1, sx+1]
            + tile[sy+1, sx-1] + 2 * tile[sy+1, sx] + tile[sy+1, sx+1]
        )

        edges[y, x] = math.sqrt(gx * gx + gy * gy)


# --------------------------------------------------
# Grid configuration
# --------------------------------------------------

def get_grid(image, threads_per_block=(16, 16)):

    height, width = image.shape

    blocks_per_grid = (
        (width + threads_per_block[0] - 1) // threads_per_block[0],
        (height + threads_per_block[1] - 1) // threads_per_block[1]
    )

    return blocks_per_grid


# --------------------------------------------------
# Benchmark using CUDA events
# --------------------------------------------------

def benchmark_kernel(kernel, d_img, d_edges, blocks, threads, runs=50):

    # Warm-up launch
    kernel[blocks, threads](d_img, d_edges)
    cuda.synchronize()

    start = cuda.event(timing=True)
    end = cuda.event(timing=True)

    start.record()

    for _ in range(runs):
        kernel[blocks, threads](d_img, d_edges)

    end.record()
    end.synchronize()

    total_time = cuda.event_elapsed_time(start, end)

    return total_time / runs


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    if not cuda.is_available():
        print("CUDA GPU is not available.")
        return

    print("GPU detected successfully!")
    print("GPU:", cuda.get_current_device().name)

    # Generate reproducible 2048 x 2048 input
    np.random.seed(42)

    image = np.random.randint(
        0, 256, size=(2048, 2048), dtype=np.uint8
    )

    # CPU reference output
    cpu_edges = sobel_cpu(image)

    # Transfer input to GPU
    img = image.astype(np.float32)
    d_img = cuda.to_device(img)

    # Allocate output arrays
    d_baseline = cuda.to_device(np.zeros_like(img))
    d_shared = cuda.to_device(np.zeros_like(img))

    threads = (16, 16)
    blocks = get_grid(image, threads)

    # Execute both kernels
    sobel_kernel[blocks, threads](d_img, d_baseline)
    sobel_shared_kernel[blocks, threads](d_img, d_shared)

    cuda.synchronize()

    # Copy results to CPU
    baseline_edges = d_baseline.copy_to_host()
    shared_edges = d_shared.copy_to_host()

    # Correctness verification
    baseline_difference = np.max(np.abs(cpu_edges - baseline_edges))
    shared_difference = np.max(np.abs(cpu_edges - shared_edges))

    print("\nImage shape:", image.shape)
    print("Threads per block:", threads)
    print("Blocks per grid:", blocks)

    print("\nCorrectness verification:")
    print("Baseline maximum difference:", baseline_difference)
    print("Shared-memory maximum difference:", shared_difference)

    baseline_correct = np.allclose(
        cpu_edges, baseline_edges, atol=1e-5
    )

    shared_correct = np.allclose(
        cpu_edges, shared_edges, atol=1e-5
    )

    print("Baseline correct:", baseline_correct)
    print("Shared-memory correct:", shared_correct)

    if not baseline_correct or not shared_correct:
        print("\nVerification failed. Do not use the timing results.")
        return

    # Benchmark both kernels with the same GPU input
    runs = 50

    baseline_time = benchmark_kernel(
        sobel_kernel, d_img, d_baseline, blocks, threads, runs
    )

    shared_time = benchmark_kernel(
        sobel_shared_kernel, d_img, d_shared, blocks, threads, runs
    )

    print("\nPerformance comparison:")
    print("Runs per kernel:", runs)
    print(f"Baseline kernel time: {baseline_time:.5f} ms")
    print(f"Shared-memory kernel time: {shared_time:.5f} ms")

    speedup = baseline_time / shared_time

    print(f"Optimization speedup: {speedup:.4f}x")

    if speedup > 1:
        print("Shared-memory version is faster.")
    elif speedup < 1:
        print("Baseline version is faster.")
    else:
        print("Both versions have equal performance.")


if __name__ == "__main__":
    main()
