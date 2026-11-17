# GTX 1080 Docker Validation Results

## 1. Purpose

This document presents the Docker-based validation of the project's three CUDA image-processing workloads on a lab PC equipped with an NVIDIA GeForce GTX 1080 GPU.

The primary objective was to verify that the existing CUDA implementations execute correctly in a containerized environment on a different GPU platform. This validation was not intended to be a complete performance comparison between the GTX 1080 and the Google Colab Tesla T4.

## 2. Experimental Environment

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce GTX 1080 |
| GPU Memory | 8 GB |
| Compute Capability | 6.1 (Pascal) |
| Host Operating System | Windows 10 |
| Container Platform | Docker Desktop 4.99.0 |
| Docker Base Image | nvidia/cuda:12.6.3-devel-ubuntu22.04 |
| Container Operating System | Ubuntu 22.04 |
| CUDA Toolkit | 12.6 |
| NVCC Version | 12.6.85 |
| Python Version | 3.10 |
| CUDA Implementation | Numba CUDA |

GPU access was verified inside the Docker container using `nvidia-smi`. Numba also successfully detected the NVIDIA GPU and reported CUDA availability.

CUDA 12.6 was selected because the GTX 1080 uses the Pascal architecture with compute capability 6.1. CUDA 13 was not used for this GPU validation.

## 3. Correctness Validation

The original CPU and GPU implementations from the project notebooks were executed inside the Docker environment.

The following tests were performed:

| Workload | Test Input | Resolution | Maximum Absolute Difference | Result |
|---|---|---|---:|---|
| Sobel Edge Detection | Real Image | 512 × 512 | 0.0 | Passed |
| Sobel Baseline vs Shared Memory | Random Image | 2048 × 2048 | 0.0 | Passed |
| Gaussian Blur | Real Image | 512 × 512 | 0.0 | Passed |
| Gaussian Blur | Random Image | 2048 × 2048 | 0.0 | Passed |
| Histogram Equalization | Real Image | 512 × 512 | 0 | Passed |
| Histogram Equalization | Random Image | 2048 × 2048 | 0 | Passed |

For Sobel and Gaussian blur, CPU and GPU outputs were compared using numerical tolerance checks. Histogram equalization outputs were compared exactly.

All tested implementations produced matching outputs.

## 4. Preliminary GPU Timing Results

The following GPU timings were recorded during the validation experiments.

| Workload | Resolution | Measurement | Time |
|---|---|---|---:|
| Sobel Baseline | 2048 × 2048 | GPU Kernel Average | 1.0721 ms |
| Sobel Shared Memory | 2048 × 2048 | GPU Kernel Average | 1.2317 ms |
| Gaussian Blur | 2048 × 2048 | GPU End-to-End | 59.0239 ms |
| Histogram Equalization | 2048 × 2048 | GPU End-to-End | 10.3237 ms |

**Note:** The Sobel measurements are averages from repeated GPU kernel executions. The Gaussian blur and histogram equalization measurements are single end-to-end validation measurements. These values have different measurement scopes and should not be directly compared.

## 5. Observations

### 5.1 Correctness Across GPU Platforms

All three CUDA workloads successfully executed on the GTX 1080 inside Docker and produced outputs matching their CPU reference implementations in the tested cases.

This provides additional evidence of the functional correctness and portability of the CUDA implementations beyond the original Colab T4 environment.

### 5.2 Sobel Baseline and Shared-Memory Comparison

For the 2048 × 2048 Sobel workload:

- Baseline kernel time: 1.0721 ms
- Shared-memory kernel time: 1.2317 ms

The shared-memory implementation was approximately 14.9% slower than the baseline implementation in this experiment.

This indicates that shared-memory optimization does not necessarily improve performance on every GPU architecture or workload configuration.

### 5.3 Histogram Equalization Warning

During histogram equalization, Numba generated a performance warning indicating that a grid size of one may result in GPU under-utilization.

This warning originated from the small single-block stages used for histogram initialization and cumulative distribution lookup-table generation.

It did not affect correctness. Both real-image and large-image tests produced exact CPU-GPU output matches.

## 6. Limitations

- The GTX 1080 experiments primarily focused on Docker compatibility and correctness validation.
- A complete repeated performance study across all resolutions and thread-block configurations was not conducted.
- Gaussian blur and histogram equalization timings are single validation measurements rather than repeated benchmark averages.
- GPU resource utilization and power measurements were not collected for this validation.
- GTX 1080 measurements are kept separate from the original Colab T4 performance results.
- A controlled performance comparison between the two GPUs would require equivalent benchmark procedures and measurement conditions.

## 7. Conclusion

The Docker environment successfully enabled GPU execution on the NVIDIA GeForce GTX 1080 using CUDA 12.6 and Numba CUDA.

Sobel edge detection, Gaussian blur, and histogram equalization all passed the tested CPU-GPU correctness checks.

The results demonstrate that the three CUDA workloads can execute correctly in a containerized environment on a second GPU platform.

The collected timing measurements provide preliminary performance observations, while a dedicated repeated benchmark would be required for a rigorous GTX 1080 versus Tesla T4 performance comparison.
