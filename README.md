# Performance, Resource Utilization, and Scalability Analysis of Cloud-Based GPU Workloads for Image Processing

## Project Details

| Attribute | Description |
|---|---|
| **Topic** | Topic 23: Performance, Resource Utilization, and Scalability Analysis of Cloud-Based GPU Workloads |
| **Domain** | GPU Computing and Image Processing |
| **Primary Workloads** | Sobel Edge Detection, Gaussian Blur, Histogram Equalization |
| **Machine Learning Application** | CNN-based Image Classification using CIFAR-10 |
| **Primary Platform** | Google Colab |
| **Primary GPU** | NVIDIA Tesla T4 |
| **Additional Validation Platform** | Docker Desktop |
| **Additional Validation GPU** | NVIDIA GeForce GTX 1080 |
| **Programming Language** | Python |
| **GPU Programming** | Numba CUDA |
| **Deep Learning Framework** | PyTorch |

## Project Overview

This project investigates the performance, resource utilization, and scalability of computationally intensive workloads executed on GPU infrastructure.

The primary focus is on evaluating GPU acceleration for image-processing operations using NVIDIA CUDA through Numba. Three image-processing workloads are implemented and analyzed:

1. **Sobel Edge Detection:** Computes image intensity gradients to identify edges.
2. **Gaussian Blur:** Applies Gaussian filtering to smooth image details and reduce noise.
3. **Histogram Equalization:** Enhances image contrast through histogram-based intensity redistribution.

Each workload is implemented using both CPU and GPU approaches. Their performance is evaluated across different image resolutions to understand how workload size influences GPU acceleration and execution overhead.

In addition to image processing, a Convolutional Neural Network (CNN) is implemented using PyTorch and trained on the CIFAR-10 dataset. This provides a machine learning application to demonstrate GPU utilization for deep learning workloads.

The primary experiments are conducted using an NVIDIA Tesla T4 GPU through Google Colab. Additional Docker-based validation is performed on an NVIDIA GeForce GTX 1080 GPU to examine the correctness and portability of the CUDA workloads across different GPU architectures and execution environments.

## Key Objectives

1. **GPU Workload Implementation:** Develop CPU and CUDA-based GPU implementations of computationally intensive image-processing operations.

2. **Performance Analysis:** Compare CPU execution time, GPU kernel execution time, and GPU end-to-end execution time across different image resolutions.

3. **Memory Transfer Analysis:** Evaluate Host-to-Device (H2D) and Device-to-Host (D2H) data transfer overhead.

4. **GPU Optimization:** Investigate thread-block configurations and shared-memory optimization to understand their impact on execution performance.

5. **Resource Utilization:** Monitor GPU utilization, memory consumption, and power usage during workload execution.

6. **Scalability Analysis:** Study how execution performance changes with increasing image resolutions and workload sizes.

7. **Machine Learning Application:** Implement and evaluate a CNN-based image classification workload using PyTorch on a GPU.

8. **Deployment and Reproducibility:** Implement Docker-based containerization to support reproducible GPU workload execution across compatible NVIDIA GPU systems.

9. **Cross-GPU Validation:** Verify the correctness of CUDA implementations on a second GPU environment using Docker.

## Workloads and Experimental Methodology

### 1. Sobel Edge Detection

The Sobel operator is used to calculate horizontal and vertical image intensity gradients for edge detection.

The experiment includes:

- CPU and GPU implementations using Numba CUDA.
- Real-image correctness verification.
- Performance comparison across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- Shared-memory optimization.
- GPU resource utilization monitoring.

### 2. Gaussian Blur

Gaussian blur applies a 5 × 5 Gaussian filter to smooth image details.

The experiment includes:

- CPU and GPU implementations.
- Real-image correctness verification.
- Performance evaluation across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- GPU resource utilization monitoring.

### 3. Histogram Equalization

Histogram equalization improves image contrast by redistributing pixel intensity values using a histogram and cumulative distribution function (CDF).

The GPU implementation uses multiple CUDA kernels for:

- Histogram initialization.
- Parallel histogram calculation.
- Cumulative distribution and lookup table generation.
- Parallel application of histogram equalization.

The experiment also evaluates correctness, execution time, memory transfer overhead, thread-block configurations, and GPU resource utilization.

### 4. Machine Learning Application: CNN on CIFAR-10

A Convolutional Neural Network is implemented using PyTorch and trained on the CIFAR-10 image classification dataset.

The experiment evaluates:

- GPU-based model training.
- Training loss across epochs.
- Training execution time.
- Classification accuracy on the test dataset.

## Experimental Environment

### 1. Primary Benchmarking Environment: Google Colab

The primary image-processing experiments were conducted using the following environment:

| Component | Configuration |
|---|---|
| Cloud Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU Memory | 15 GB |
| NVIDIA Driver | 580.82.07 |
| CUDA Version | 13.0 |
| Python | Python 3 |
| GPU Programming | Numba CUDA |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Visualization | Matplotlib |
| Image Processing | scikit-image |
| Machine Learning | PyTorch and Torchvision |

The actual software versions and hardware availability may vary across cloud runtime sessions.

### 2. Docker-Based Validation Environment: NVIDIA GTX 1080

An additional validation environment was configured using Docker Desktop on a Windows 10 system with an NVIDIA GeForce GTX 1080 GPU.

| Component | Configuration |
|---|---|
| Host Operating System | Windows 10 |
| Container Platform | Docker Desktop 4.99.0 |
| Container Operating System | Ubuntu 22.04 |
| GPU | NVIDIA GeForce GTX 1080 |
| GPU Memory | 8 GB |
| GPU Architecture | Pascal |
| Compute Capability | 6.1 |
| NVIDIA Driver | 582.53 |
| CUDA Toolkit | 12.6 |
| Base Docker Image | nvidia/cuda:12.6.3-devel-ubuntu22.04 |
| Python | Python 3.10 |
| GPU Programming | Numba CUDA |
| Numerical Computing | NumPy |

Docker GPU passthrough was verified successfully using NVIDIA's container runtime support. The GTX 1080 was detected inside the container, and CUDA availability was confirmed through Numba.

The Docker environment was used for additional correctness validation of all three image-processing workloads and preliminary timing observations. It was not intended to replace the primary Colab benchmarking environment.

## Docker-Based Cross-GPU Validation

The CUDA implementations were additionally executed inside a Docker container using an NVIDIA GTX 1080 GPU.

The purpose of this validation was to verify that the image-processing implementations could execute correctly in a separate containerized environment with a different GPU architecture.

The validation included:

- Docker image creation using an NVIDIA CUDA development base image.
- NVIDIA GPU passthrough verification.
- CUDA toolkit and Numba CUDA availability checks.
- Execution of the Sobel edge detection implementation.
- Execution of the Gaussian blur implementation.
- Execution of the histogram equalization implementation.
- CPU-GPU output comparison for correctness verification.
- Preliminary timing observations for the GPU implementations.

### Correctness Validation

All three image-processing workloads successfully passed the correctness checks on the GTX 1080.

| Workload | Validation Input | Resolution | Maximum Absolute Difference | Status |
|---|---|---:|---:|---|
| Sobel Edge Detection | Real Image | 512 × 512 | 0.0 | Passed |
| Sobel Baseline vs Shared Memory | Random Image | 2048 × 2048 | 0.0 | Passed |
| Gaussian Blur | Real Image | 512 × 512 | 0.0 | Passed |
| Gaussian Blur | Random Image | 2048 × 2048 | 0.0 | Passed |
| Histogram Equalization | Real Image | 512 × 512 | 0 | Passed |
| Histogram Equalization | Random Image | 2048 × 2048 | 0 | Passed |

### Preliminary Timing Observations

The following timing measurements were obtained during the GTX 1080 validation:

| Workload | Measurement | Execution Time |
|---|---|---:|
| Sobel Baseline | Average GPU kernel time (30 runs) | 1.0721 ms |
| Sobel Shared Memory | Average GPU kernel time (30 runs) | 1.2317 ms |
| Gaussian Blur | Single GPU end-to-end measurement | 59.0239 ms |
| Histogram Equalization | Single GPU end-to-end measurement | 10.3237 ms |

For the Sobel workload, the shared-memory implementation achieved a speedup of approximately 0.87× relative to the baseline, indicating that it was slower in this particular preliminary GTX 1080 measurement.

The Gaussian blur and histogram equalization timings represent single end-to-end measurements, including their respective execution overheads.

**Note:** These GTX 1080 measurements are preliminary validation observations. They should not be directly compared with the repeated Colab T4 benchmark measurements because the measurement procedures and execution environments differ. A complete GTX 1080 performance, scalability, and resource-utilization study was not conducted.

## Performance Evaluation

The primary Colab T4 workloads are evaluated using the following performance metrics:

| Metric | Purpose |
|---|---|
| CPU Execution Time | Measures CPU processing time |
| GPU Kernel Time | Measures GPU computation time excluding host-device transfers |
| GPU End-to-End Time | Measures the complete GPU execution process |
| Kernel Speedup | Compares CPU execution time with GPU kernel time |
| End-to-End Speedup | Compares CPU execution time with complete GPU execution time |
| H2D Transfer Time | Measures CPU-to-GPU data transfer time |
| D2H Transfer Time | Measures GPU-to-CPU data transfer time |
| GPU Utilization | Measures GPU processing activity |
| GPU Memory Usage | Measures GPU memory consumption |
| Power Consumption | Observes GPU power usage |

Image resolutions of 256 × 256, 512 × 512, 1024 × 1024, and 2048 × 2048 are used for the primary performance comparisons.

The Colab T4 environment is used for the main performance, scalability, optimization, and resource-utilization analysis. The GTX 1080 environment provides additional cross-GPU correctness validation and preliminary timing observations.

## Repository Structure

```text
cloud-gpu-image-benchmarking/
│
├── README.md
│
├── project/
│   ├── problem-formulation.md
│   ├── literature-review.md
│   └── workload-selection.md
│
├── docker/
│   └── Dockerfile
│
├── docs/
│   ├── cuda-fundamentals.md
│   └── cloud-deployment.md
│
├── notebooks/
│   ├── cuda_vector_add.ipynb
│   ├── sobel_gpu_experiments.ipynb
│   ├── gaussian_blur_experiments.ipynb
│   ├── histogram_equalization_experiments.ipynb
│   └── gpu_ml_application.ipynb
│
└── results/
    ├── figures/
    │
    └── docker-gtx1080/
        ├── correctness-results.md
        └── validation-summary.csv
```

## Project Deliverables

- **GPU Workload Implementations:** CPU and CUDA-based implementations of Sobel edge detection, Gaussian blur, and histogram equalization.
- **Machine Learning Application:** CNN implementation and evaluation using the CIFAR-10 dataset.
- **Experimental Notebooks:** Self-contained Jupyter notebooks containing implementations, performance experiments, correctness checks, and visualizations.
- **Performance Analysis:** Comparative evaluation of CPU and GPU execution times, speedups, memory transfers, and thread-block configurations.
- **Resource Profiling:** GPU utilization, memory consumption, and power measurements.
- **Visual Analytics:** Graphs illustrating workload performance and resource utilization.
- **Docker Validation:** Containerized execution and cross-GPU correctness validation using an NVIDIA GTX 1080.
- **Deployment Documentation:** Cloud execution, Docker configuration, GPU passthrough, and reproducibility instructions.
- **Academic Report:** A technical report documenting the methodology, experimental findings, limitations, and conclusions.

## Deployment and Reproducibility

The project uses two separate GPU execution environments.

**Google Colab:** The primary benchmarking environment, using an NVIDIA Tesla T4 GPU for image-processing performance evaluation, scalability experiments, optimization analysis, and resource monitoring.

**Docker:** An additional containerized environment configured with an NVIDIA GTX 1080 GPU. Docker GPU passthrough was verified, and all three image-processing workloads were successfully executed and checked for correctness.

The Docker environment used the NVIDIA CUDA 12.6 development image with Ubuntu 22.04 and Numba CUDA.

The Docker validation demonstrates the ability to execute the CUDA workloads in a separate containerized GPU environment. However, the primary performance conclusions are based on the Colab T4 experiments.

The Docker configuration is maintained in [`docker/Dockerfile`](docker/Dockerfile), while the complete deployment and execution procedure is documented in [`docs/cloud-deployment.md`](docs/cloud-deployment.md).

## Research and References

1. Kirk, D. B., Hwu, W. W., & El Hajj, I. *Programming Massively Parallel Processors: A Hands-on Approach.* Morgan Kaufmann.

2. NVIDIA Corporation. *CUDA C++ Best Practices Guide.*

3. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems.

4. Harris, M. (2007). *Optimizing Parallel Reduction in CUDA.* NVIDIA Developer Technology.

5. NVIDIA Corporation. *CUDA C++ Programming Guide.*

6. Numba Development Team. *Numba Documentation: CUDA Support.* https://numba.readthedocs.io/

7. Gonzalez, R. C., & Woods, R. E. *Digital Image Processing.* Pearson.

8. Szeliski, R. *Computer Vision: Algorithms and Applications.* Springer.
