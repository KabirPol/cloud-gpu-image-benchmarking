# Workload Selection and Experimental Design

## 1. Purpose

This project evaluates the performance, resource utilization, and scalability of image-processing workloads executed on CPU and GPU platforms, including a cloud-based GPU environment.

The selected workload should be computationally intensive, parallelizable, easy to validate, and suitable for testing with different image sizes and batch sizes.

## 2. Candidate Workloads

| Workload | Description | Suitability |
|---|---|---|
| Grayscale conversion | Converts a color image into a single-channel image | Simple and useful as a baseline, but relatively low computational intensity |
| Gaussian blur | Smooths an image using a Gaussian filter | Parallelizable and suitable for testing memory access and computation |
| Sobel edge detection | Detects image edges by computing image intensity gradients | Parallelizable, computationally meaningful, and suitable for CPU/GPU comparison |

## 3. Selected Workload: Sobel Edge Detection

Sobel edge detection is selected as the primary workload. It computes horizontal and vertical intensity gradients for each pixel using two convolution filters.

The Sobel filters are:

\[
S_x =
\begin{bmatrix}
-1 & 0 & 1 \\
-2 & 0 & 2 \\
-1 & 0 & 1
\end{bmatrix}
\qquad
S_y =
\begin{bmatrix}
-1 & -2 & -1 \\
0 & 0 & 0 \\
1 & 2 & 1
\end{bmatrix}
\]

For an input image \(I\), the gradients are:

\[
G_x = I * S_x
\]

\[
G_y = I * S_y
\]

where \(*\) denotes convolution. The gradient magnitude is calculated as:

\[
G = \sqrt{G_x^2 + G_y^2}
\]

The output represents the strength of edges in the image.

### Reasons for Selection

- The computation can be performed independently for many pixels, making it suitable for parallel execution.
- The same mathematical operation can be implemented on CPU and GPU.
- Image dimensions and batch sizes can be varied systematically.
- The output can be compared across implementations to check correctness.
- Execution time, throughput, and resource utilization can be measured.

## 4. CPU and GPU Implementations

A CPU implementation will serve as the baseline. A GPU implementation will perform the same Sobel operation using GPU parallelism.

To make the comparison meaningful, both implementations should use:

- The same input images and image dimensions.
- The same filter coefficients and boundary-handling rule.
- The same output representation and precision, where practical.
- A correctness check before performance measurements.

The implementation should document whether timing includes only the computation or also includes data transfer and preprocessing.

## 5. Experimental Parameters

### Image Resolutions

The following image sizes are proposed:

| Image size | Number of pixels |
|---|---:|
| 256 × 256 | 65,536 |
| 512 × 512 | 262,144 |
| 1024 × 1024 | 1,048,576 |
| 2048 × 2048 | 4,194,304 |

These sizes provide a range of workload scales, from small images to larger images with more pixel-level operations.

### Batch Sizes

The proposed batch sizes are:

| Batch size |
|---:|
| 1 |
| 4 |
| 8 |
| 16 |

Batch size indicates the number of images processed in one workload run. If a batch size cannot be supported by a particular environment, the limitation should be recorded rather than silently changing the experiment.

## 6. Performance and Resource Metrics

The experiment will record the following metrics where the execution environment makes them available.

| Metric | Definition |
|---|---|
| Kernel execution time | Time spent executing the Sobel computation on the device |
| End-to-end execution time | Total time including relevant data transfer and processing overhead |
| Throughput | Number of images processed per unit time |
| Speedup | CPU execution time divided by GPU execution time for the same workload |
| GPU utilization | Fraction or percentage of GPU activity reported by the environment |
| GPU memory usage | Device memory consumed during execution |

Kernel time and end-to-end time should be reported separately when both can be measured. This distinction is important because data transfer and setup overhead may affect the overall result.

## 7. Experimental Procedure

1. Prepare a fixed set of input images or reproducibly generated images.
2. Implement Sobel edge detection on the CPU.
3. Implement the equivalent Sobel operation on the GPU.
4. Compare the outputs to verify that the implementations produce sufficiently similar results.
5. Run the workload for each selected image resolution.
6. Repeat the runs for each selected batch size.
7. Perform warm-up runs where required by the environment.
8. Repeat each measurement several times and report a summary statistic, such as the mean or median.
9. Record the environment and configuration used for every experiment.
10. Plot the measured performance and resource metrics for analysis.

No performance outcome should be assumed in advance. The results section should be based on measurements collected during the experiments.

## 8. Result Presentation

The results can be presented using the following figures:

- CPU and GPU execution time versus image resolution.
- CPU and GPU execution time versus batch size.
- GPU speedup versus image resolution.
- Throughput versus batch size.
- GPU utilization and memory usage versus workload size, when those metrics are available.
- Local and cloud execution comparisons, when comparable configurations can be measured.

Each figure should state the workload parameters and the metric being shown. The report should discuss observed trends and possible explanations while distinguishing measured evidence from interpretation.

## 9. Reproducibility

For reproducibility, record:

- Hardware and cloud instance details.
- Operating system and relevant software versions.
- GPU model and available device memory, if exposed.
- Image dimensions and batch size.
- Number of repetitions and warm-up policy.
- Timing method and whether data transfer is included.
- Any constraints, errors, or unavailable monitoring metrics.

The experiment should use the same workload definition across platforms as closely as possible. Differences in hardware, software, and measurement access should be documented.
