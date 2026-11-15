# Consolidated Performance and Scalability Analysis

## 1. Introduction

This document presents the consolidated experimental results obtained from the CPU and GPU implementations of three image-processing workloads:

1. Sobel Edge Detection
2. Gaussian Blur
3. Histogram Equalization

The experiments evaluate execution performance, GPU acceleration, memory transfer overhead, thread-block configurations, and GPU resource utilization using an NVIDIA Tesla T4 GPU on Google Colab.

The objective is to understand how different image-processing workloads behave under GPU acceleration and how their performance changes with increasing image resolution.

## 2. Experimental Setup

### 2.1 Hardware and Software Environment

| Component | Configuration |
|---|---|
| Cloud Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 15 GB |
| NVIDIA Driver | 580.82.07 |
| CUDA Version | 13.0 |
| GPU Programming | Numba CUDA |
| Numerical Computing | NumPy |
| Performance Analysis | Pandas |
| Visualization | Matplotlib |
| Image Processing | scikit-image |

### 2.2 Workload Configuration

All three image-processing workloads were evaluated using grayscale images at four different resolutions.

| Parameter | Configuration |
|---|---|
| Image Type | Grayscale |
| Image Format | uint8 input |
| Image Resolutions | 256 × 256, 512 × 512, 1024 × 1024, 2048 × 2048 |
| Primary Test Image | `skimage.data.camera()` |
| CPU Implementation | Python with Numba JIT where applicable |
| GPU Implementation | Numba CUDA |

The same input image and corresponding processing parameters were used for CPU and GPU correctness comparisons.

### 2.3 Performance Metrics

The following metrics were considered:

- **CPU Execution Time:** Time required to execute the workload on the CPU.
- **GPU Kernel Execution Time:** Time spent executing GPU computation kernels.
- **GPU End-to-End Execution Time:** Total GPU execution time, including relevant memory transfers and execution overhead.
- **Kernel Speedup:** Ratio of CPU execution time to GPU kernel execution time.
- **End-to-End Speedup:** Ratio of CPU execution time to GPU end-to-end execution time.
- **GPU Utilization:** Percentage of GPU processing activity observed during workload execution.
- **GPU Memory Usage:** GPU memory consumption during execution.

Speedup is calculated using the following formulas:

- **Kernel Speedup** = `CPU Execution Time / GPU Kernel Execution Time`
- **End-to-End Speedup** = `CPU Execution Time / GPU End-to-End Execution Time`

A speedup greater than 1 indicates GPU acceleration, while a value below 1 indicates that the CPU implementation is faster for the measured execution scope.

## 3. Sobel Edge Detection Results

Sobel edge detection computes horizontal and vertical intensity gradients to identify edges in an image.

### 3.1 CPU and GPU Performance Comparison

| Resolution | CPU Time (ms) | GPU Kernel (ms) | GPU End-to-End (ms) | Kernel Speedup | End-to-End Speedup |
|---|---:|---:|---:|---:|---:|
| 256 × 256 | 1.0820 | 0.0666 | 0.9631 | 16.24× | 1.12× |
| 512 × 512 | 4.4035 | 0.2100 | 1.7499 | 20.97× | 2.52× |
| 1024 × 1024 | 23.9227 | 0.8161 | 5.1526 | 29.31× | 4.64× |
| 2048 × 2048 | 118.4407 | 3.2279 | 11.6378 | 36.69× | 10.18× |

### 3.2 Observations

- GPU kernel execution is faster than CPU execution at every tested resolution.
- Kernel speedup increases with image resolution, reaching approximately 36.69× at 2048 × 2048.
- End-to-end GPU execution also becomes increasingly beneficial as image resolution increases.
- The difference between kernel and end-to-end performance demonstrates the impact of memory transfers and execution overhead.

### 3.3 Thread-Block Configuration

| Thread Block | GPU Kernel Time (ms) | Correctness |
|---|---:|---|
| 8 × 8 | 0.21389 | Passed |
| 16 × 16 | 0.22871 | Passed |
| 32 × 32 | 0.23253 | Passed |

The tested thread-block configurations produced correct outputs. The 8 × 8 configuration achieved the lowest measured execution time in this experiment.

### 3.4 Shared-Memory Optimization

| Implementation | GPU Kernel Time (ms) |
|---|---:|
| Baseline Global-Memory Implementation | 3.22748 |
| Shared-Memory Implementation | 1.80614 |

The shared-memory implementation achieved approximately 1.79× speedup over the baseline implementation.

Both implementations produced matching outputs.

### 3.5 GPU Resource Utilization

| Metric | Measured Value |
|---|---:|
| Image Resolution | 2048 × 2048 |
| Number of Executions | 100 |
| Total Monitoring Duration | 2.438 s |
| Mean GPU Utilization | 27.07% |
| Mean GPU Memory Usage | 328.43 MB |
| Mean Power Consumption | 29.26 W |

## 4. Gaussian Blur Results

Gaussian blur applies a 5 × 5 Gaussian filter to smooth image details and reduce noise.

### 4.1 CPU and GPU Performance Comparison

| Resolution | CPU Time (ms) | GPU Kernel (ms) | GPU End-to-End (ms) | Kernel Speedup | End-to-End Speedup |
|---|---:|---:|---:|---:|---:|
| 256 × 256 | 9.5561 | 0.1737 | 2.8541 | 55.02× | 3.35× |
| 512 × 512 | 57.0255 | 0.3454 | 2.4261 | 165.09× | 23.50× |
| 1024 × 1024 | 195.7536 | 1.3303 | 6.0277 | 147.15× | 32.48× |
| 2048 × 2048 | 274.2748 | 4.5730 | 10.4879 | 59.98× | 26.15× |

### 4.2 Observations

- GPU kernel execution is faster than CPU execution at all tested resolutions.
- The highest kernel speedup was approximately 165.09× at 512 × 512.
- The highest end-to-end speedup was approximately 32.48× at 1024 × 1024.
- The speedup does not increase continuously with resolution. This can be influenced by memory behavior, execution overhead, and measurement variability.
- GPU acceleration provides substantial benefits for this computationally intensive filtering operation.

### 4.3 Thread-Block Configuration

| Thread Block | GPU Kernel Time (ms) | Correctness |
|---|---:|---|
| 8 × 8 | 0.13254 | Passed |
| 16 × 16 | 0.13524 | Passed |
| 32 × 32 | 0.26464 | Passed |

The 8 × 8 configuration achieved the lowest measured execution time among the tested configurations.

### 4.4 GPU Resource Utilization

| Metric | Measured Value |
|---|---:|
| Image Resolution | 2048 × 2048 |
| Number of Executions | 100 |
| Total Monitoring Duration | 1.72 s |
| Mean GPU Utilization | 38.55% |
| Mean GPU Memory Usage | 209.36 MB |
| Mean Power Consumption | 31.94 W |
| GPU Utilization Standard Deviation | 12.33% |
| GPU Memory Standard Deviation | 35.12 MB |
| Power Standard Deviation | 1.99 W |

## 5. Histogram Equalization Results

Histogram equalization enhances image contrast by redistributing pixel intensity values using a histogram and cumulative distribution function.

The GPU implementation uses multiple CUDA kernels for histogram initialization, histogram calculation, CDF and lookup table generation, and output image generation.

### 5.1 CPU and GPU Performance Comparison

| Resolution | CPU Time (ms) | GPU Kernel (ms) | GPU End-to-End (ms) | Kernel Speedup | End-to-End Speedup |
|---|---:|---:|---:|---:|---:|
| 256 × 256 | 0.1156 | 0.2433 | 1.1657 | 0.48× | 0.10× |
| 512 × 512 | 0.3268 | 0.3590 | 1.2331 | 0.91× | 0.27× |
| 1024 × 1024 | 1.3094 | 0.9014 | 2.1927 | 1.45× | 0.60× |
| 2048 × 2048 | 4.9937 | 3.0652 | 5.5428 | 1.63× | 0.90× |

### 5.2 Observations

- At smaller resolutions, CPU execution is faster than GPU execution.
- GPU kernel execution becomes faster than CPU execution at 1024 × 1024 and 2048 × 2048.
- The highest kernel speedup is approximately 1.63× at 2048 × 2048.
- End-to-end GPU execution remains slower than CPU execution at all tested resolutions.
- The histogram equalization pipeline requires multiple kernel launches and intermediate operations. Consequently, memory allocation, transfers, and launch overhead have a significant impact on total execution time.
- This workload demonstrates that GPU acceleration at the kernel level does not necessarily translate into better end-to-end performance.

### 5.3 Thread-Block Configuration

| Thread Block | GPU Kernel Time (ms) | Correctness |
|---|---:|---|
| 8 × 8 | 0.22345 | Passed |
| 16 × 16 | 0.22837 | Passed |
| 32 × 32 | 0.25626 | Passed |

The 8 × 8 configuration achieved the lowest measured execution time among the tested configurations.

### 5.4 GPU Resource Utilization

| Metric | Measured Value |
|---|---:|
| Image Resolution | 2048 × 2048 |
| Number of Executions | 100 |
| Total Monitoring Duration | 0.50 s |
| Mean GPU Utilization | 29.50% |
| Mean GPU Memory Usage | 132 MB |
| Mean Power Consumption | 37.86 W |
| Number of Monitoring Samples | 4 |

The resource monitoring interval was relatively short and produced only four samples. Therefore, these measurements should be treated as indicative observations rather than comprehensive sustained-utilization statistics.

## 6. Comparative Analysis of Image-Processing Workloads

### 6.1 Performance Comparison at 2048 × 2048

| Workload | CPU Time (ms) | GPU Kernel (ms) | GPU End-to-End (ms) | Kernel Speedup | End-to-End Speedup |
|---|---:|---:|---:|---:|---:|
| Sobel | 118.4407 | 3.2279 | 11.6378 | 36.69× | 10.18× |
| Gaussian Blur | 274.2748 | 4.5730 | 10.4879 | 59.98× | 26.15× |
| Histogram Equalization | 4.9937 | 3.0652 | 5.5428 | 1.63× | 0.90× |

### 6.2 Comparative Observations

- Gaussian blur achieved the highest kernel speedup among the three workloads at 2048 × 2048.
- Gaussian blur also achieved the highest end-to-end speedup at this resolution.
- Sobel edge detection demonstrated strong GPU acceleration, with an end-to-end speedup of approximately 10.18×.
- Histogram equalization achieved a smaller kernel-level performance improvement and did not achieve end-to-end acceleration.
- The results demonstrate that GPU performance depends on workload characteristics, computational complexity, memory operations, and execution overhead.

### 6.3 Overall Findings

The experiments demonstrate that GPU acceleration is particularly beneficial for computationally intensive image-processing operations such as Sobel edge detection and Gaussian blur.

Histogram equalization exhibits a different performance pattern because its multi-stage processing pipeline introduces additional execution overhead.

Therefore, selecting a GPU implementation should consider complete execution time rather than kernel execution time alone.

## 7. Correctness Verification

All three image-processing workloads were tested by comparing their CPU and GPU outputs.

| Workload | Real Image Correctness | Synthetic Resolution Tests |
|---|---|---|
| Sobel Edge Detection | Passed | Passed |
| Gaussian Blur | Passed | Passed |
| Histogram Equalization | Passed | Passed |

The implementations produced matching CPU and GPU outputs in the reported correctness tests.

## 8. Limitations

The following limitations should be considered when interpreting the experimental results:

1. Experiments were conducted using a shared cloud GPU environment, where GPU availability and background activity may vary.
2. Execution timings can be affected by runtime overhead, GPU initialization, memory allocation, and measurement variability.
3. GPU utilization and memory measurements represent sampled observations rather than continuous hardware profiling.
4. Histogram equalization resource monitoring produced only four samples, limiting the reliability of sustained-utilization conclusions.
5. The experiments use a limited set of image-processing operations and resolutions.
6. The results represent a single NVIDIA Tesla T4 configuration and should not be generalized to all GPU architectures.

## 9. Conclusion

The experimental analysis demonstrates that cloud-based GPU acceleration can significantly improve the execution performance of suitable image-processing workloads.

Sobel edge detection and Gaussian blur achieved substantial GPU speedups, particularly at larger image resolutions. In contrast, histogram equalization demonstrated that multi-stage GPU processing pipelines may experience substantial overhead, limiting end-to-end acceleration even when GPU kernel execution becomes faster than CPU execution.

The results emphasize the importance of evaluating kernel execution, memory transfer overhead, end-to-end execution time, and GPU resource utilization together when analyzing GPU workload performance and scalability.

Further work will focus on Docker-based GPU deployment, reproducibility, and final consolidated project evaluation.
