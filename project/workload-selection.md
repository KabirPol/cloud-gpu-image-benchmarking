# Workload Selection and Experimental Design

## 1. Purpose

This project evaluates the performance, resource utilization, and scalability of image-processing workloads executed on CPU and GPU platforms, with particular emphasis on cloud-based GPU execution.

The selected workloads should be computationally meaningful, parallelizable, suitable for CPU–GPU comparison, and capable of being evaluated using different image sizes and workload configurations.

Sobel edge detection is selected as the primary workload for detailed performance analysis. Gaussian blur and histogram equalization are included as additional workloads to evaluate GPU acceleration across different types of image-processing operations.

## 2. Candidate and Selected Workloads

The following image-processing operations are considered:

| Workload | Description | Role in the project |
|---|---|---|
| Sobel edge detection | Detects image edges by computing horizontal and vertical intensity gradients | Primary workload |
| Gaussian blur | Smooths an image using a Gaussian filter | Additional workload |
| Histogram equalization | Improves image contrast by redistributing intensity values using a histogram | Additional workload |

### Reasons for Selection

- All three operations are commonly used in image processing.
- Their computations can be parallelized using GPU processing.
- CPU and GPU implementations can be compared using equivalent input images.
- They represent different computational characteristics: convolution-based filtering, gradient computation, and histogram-based processing.
- Their execution time and resource requirements can be studied across different image sizes.

## 3. Primary Workload: Sobel Edge Detection

Sobel edge detection is selected as the primary workload. It computes horizontal and vertical intensity gradients for each pixel using two convolution filters.

The Sobel filters are:

**Horizontal filter (Sx):**

```text
-1   0   1
-2   0   2
-1   0   1
```

**Vertical filter (Sy):**

```text
-1  -2  -1
 0   0   0
 1   2   1
```

For an input image I, the horizontal and vertical gradients are calculated as:

```text
Gx = I * Sx
Gy = I * Sy
```

Here, `*` denotes the convolution operation.

The gradient magnitude is calculated as:

```text
G = sqrt(Gx² + Gy²)
```

The output represents the strength of edges in the image.

### Reasons for Selecting Sobel as the Primary Workload

- The computation can be performed independently for many pixels, making it suitable for parallel execution.
- The same mathematical operation can be implemented on CPU and GPU.
- Image dimensions can be varied systematically to study performance.
- The output can be compared across implementations to check correctness.
- Different thread-block configurations can be evaluated.
- Kernel optimization techniques, such as shared memory, can be investigated.
- Execution time, speedup, and GPU resource utilization can be measured.

## 4. Additional Workloads

### 4.1 Gaussian Blur

Gaussian blur is an image-filtering operation that reduces noise and smooths image details using a Gaussian kernel.

The operation applies a weighted convolution to neighboring pixels. Pixels closer to the center of the kernel receive higher weights than pixels farther away.

Gaussian blur is included to evaluate GPU performance for another convolution-based image-processing operation and to compare its computational behavior with Sobel edge detection.

### 4.2 Histogram Equalization

Histogram equalization is an image-enhancement technique that improves image contrast by redistributing pixel intensity values.

It uses the image histogram and cumulative distribution function to calculate a mapping from the original intensity values to new intensity values.

This operation is included to investigate GPU performance for a histogram-based workload, which has different computational characteristics from convolution-based operations.

## 5. CPU and GPU Implementations

CPU implementations will serve as the baseline, while GPU implementations will perform equivalent image-processing operations using GPU parallelism.

The implementations will use Python, NumPy, and CUDA through Numba.

Google Colab with an NVIDIA Tesla T4 GPU is used as the cloud-based GPU execution environment.

To make CPU–GPU comparisons meaningful, both implementations should use:

- The same input images and image dimensions.
- Equivalent processing operations and algorithm parameters.
- Consistent boundary-handling rules where applicable.
- Comparable output representations and numerical precision.
- Correctness checks before performance measurements.

The implementation should document whether timing includes only computation or also includes data transfer, memory allocation, and other processing overhead.

## 6. Experimental Parameters

### 6.1 Image Resolutions

The following image sizes are selected for performance evaluation:

| Image size | Number of pixels |
|---|---:|
| 256 × 256 | 65,536 |
| 512 × 512 | 262,144 |
| 1024 × 1024 | 1,048,576 |
| 2048 × 2048 | 4,194,304 |

These sizes provide a range of workload scales, from small images to larger images with more pixel-level operations.

The same image resolutions will be used for the additional workloads where practical.

### 6.2 Batch Sizes

The following batch sizes are proposed for evaluating the effect of processing multiple images:

| Batch size |
|---:|
| 1 |
| 4 |
| 8 |
| 16 |

Batch size indicates the number of images processed in one workload run.

Batch-size experiments will be conducted where time and available GPU resources permit. If a configuration cannot be tested, the limitation will be documented.

### 6.3 GPU Thread-Block Configurations

For the primary Sobel workload, different GPU thread-block configurations are evaluated to study their influence on kernel execution time.

The configurations considered are:

| Threads per block | Total threads |
|---|---:|
| 8 × 8 | 64 |
| 16 × 16 | 256 |
| 32 × 32 | 1024 |

The configurations are compared using the same input workload and GPU environment.

### 6.4 Kernel Optimization

The Sobel workload is also used to investigate shared-memory optimization.

The baseline GPU implementation accesses image data from global memory. The optimized implementation uses shared memory to reuse neighboring pixel values within a thread block.

The two implementations will be compared using:

- Output correctness.
- GPU kernel execution time.
- Relative performance improvement or overhead.

The effectiveness of the optimization will be determined from measured results rather than assumed in advance.

## 7. Performance and Resource Metrics

The experiments will record the following metrics where the execution environment makes them available.

| Metric | Definition |
|---|---|
| CPU execution time | Time required by the CPU implementation |
| GPU kernel execution time | Time spent executing the computation on the GPU |
| End-to-end execution time | Total processing time, including relevant data transfers and overhead |
| Throughput | Number of images processed per unit of time |
| Speedup | Ratio of CPU execution time to GPU execution time for equivalent work |
| GPU utilization | Percentage of GPU activity reported by the monitoring tools |
| GPU memory usage | Device memory consumed during workload execution |
| GPU power consumption | Power usage reported by GPU monitoring tools |
| Data-transfer overhead | Time associated with transferring data between host and GPU memory |
| Scalability | How performance changes as workload size or batch size increases |

Kernel execution time and end-to-end execution time will be reported separately when both can be measured.

Resource measurements will be interpreted according to the monitoring capabilities and sampling limitations of the cloud environment.

## 8. Experimental Procedure

The experimental procedure is organized as follows:

1. Prepare reproducible input images for the selected image resolutions.
2. Implement the primary Sobel workload on the CPU.
3. Implement the equivalent Sobel operation on the GPU.
4. Compare CPU and GPU outputs to verify correctness.
5. Measure CPU and GPU execution times across the selected image resolutions.
6. Evaluate different GPU thread-block configurations for Sobel.
7. Implement and evaluate the shared-memory Sobel kernel against the baseline implementation.
8. Monitor GPU utilization, memory consumption, and power usage during selected workloads.
9. Implement Gaussian blur and histogram equalization on the CPU and GPU.
10. Compare the performance of the additional workloads across suitable image sizes.
11. Conduct batch-size experiments where feasible.
12. Repeat measurements, record the execution environment, and summarize the results.

Warm-up runs will be performed where required. Repeated measurements will be used to reduce the influence of temporary variations in execution conditions.

## 9. Result Presentation

The results will be presented using tables and graphs, including:

- CPU and GPU execution time versus image resolution.
- CPU and GPU end-to-end execution time versus image resolution.
- GPU kernel speedup versus image resolution.
- Performance comparison across GPU thread-block configurations.
- Baseline versus shared-memory Sobel kernel execution time.
- Throughput versus batch size, where measured.
- GPU utilization, memory usage, and power consumption during selected workloads.
- Performance comparisons between Sobel, Gaussian blur, and histogram equalization.

Each figure will state the workload parameters and the metric being shown. The report will discuss observed trends and possible explanations while distinguishing measured evidence from interpretation.

## 10. Reproducibility

For reproducibility, the following information will be recorded:

- Hardware and cloud environment details.
- Operating system and relevant software versions.
- GPU model and available device memory.
- Image dimensions and batch size.
- Number of repetitions and warm-up policy.
- Thread-block configuration, where applicable.
- Timing method and whether data transfer is included.
- GPU monitoring method and sampling interval.
- Any constraints, errors, or unavailable monitoring metrics.

The experiment will use consistent workload definitions across CPU and GPU implementations as closely as possible. Differences in hardware, software, and measurement access will be documented.

The source code, notebooks, experimental measurements, and generated figures will be maintained in the project repository to support reproducibility.
