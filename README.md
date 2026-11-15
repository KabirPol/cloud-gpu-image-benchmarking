# Performance, Resource Utilization, and Scalability Analysis of Cloud-Based GPU Workloads for Image Processing

## Project Details

| Attribute | Description |
|---|---|
| **Topic** | Topic 23: Performance, Resource Utilization, and Scalability Analysis of Cloud-Based GPU Workloads |
| **Domain** | GPU Computing and Image Processing |
| **Primary Workloads** | Sobel Edge Detection, Gaussian Blur, Histogram Equalization |
| **Machine Learning Application** | CNN-based Image Classification using CIFAR-10 |
| **Cloud Platform** | Google Colab |
| **GPU** | NVIDIA Tesla T4 |
| **Programming Language** | Python |
| **GPU Programming** | Numba CUDA |
| **Deep Learning Framework** | PyTorch |

## Project Overview

This project investigates the performance, resource utilization, and scalability of computationally intensive workloads executed on cloud-based GPU infrastructure.

The primary focus is on evaluating GPU acceleration for image-processing operations using NVIDIA CUDA through Numba. Three image-processing workloads are implemented and analyzed:

1. **Sobel Edge Detection:** Computes image intensity gradients to identify edges.
2. **Gaussian Blur:** Applies Gaussian filtering to smooth image details and reduce noise.
3. **Histogram Equalization:** Enhances image contrast through histogram-based intensity redistribution.

Each workload is implemented using both CPU and GPU approaches. Their performance is evaluated across different image resolutions to understand how workload size influences GPU acceleration and execution overhead.

In addition to image processing, a Convolutional Neural Network (CNN) is implemented using PyTorch and trained on the CIFAR-10 dataset. This provides a machine learning application to demonstrate GPU utilization for deep learning workloads.

The experiments are conducted using an NVIDIA Tesla T4 GPU through Google Colab.

## Key Objectives

1. **GPU Workload Implementation:** Develop CPU and CUDA-based GPU implementations of computationally intensive image-processing operations.

2. **Performance Analysis:** Compare CPU execution time, GPU kernel execution time, and GPU end-to-end execution time across different image resolutions.

3. **Memory Transfer Analysis:** Evaluate Host-to-Device (H2D) and Device-to-Host (D2H) data transfer overhead.

4. **GPU Optimization:** Investigate thread-block configurations and shared-memory optimization to understand their impact on execution performance.

5. **Resource Utilization:** Monitor GPU utilization, memory consumption, and power usage during workload execution.

6. **Scalability Analysis:** Study how execution performance changes with increasing image resolutions and workload sizes.

7. **Machine Learning Application:** Implement and evaluate a CNN-based image classification workload using PyTorch on a cloud GPU.

8. **Deployment and Reproducibility:** Explore Docker-based containerization to support reproducible GPU workload execution.

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

## Performance Evaluation

The workloads are evaluated using the following performance metrics:

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

## Repository Structure

```text
cloud-gpu-workload-analysis/
│
├── README.md
│
├── project/
│   ├── problem-formulation.md
│   ├── literature-review.md
│   └── workload-selection.md
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
    └── figures/
```

## Project Deliverables

- **GPU Workload Implementations:** CPU and CUDA-based implementations of Sobel edge detection, Gaussian blur, and histogram equalization.
- **Machine Learning Application:** CNN implementation and evaluation using the CIFAR-10 dataset.
- **Experimental Notebooks:** Self-contained Jupyter notebooks containing implementations, performance experiments, correctness checks, and visualizations.
- **Performance Analysis:** Comparative evaluation of CPU and GPU execution times, speedups, memory transfers, and thread-block configurations.
- **Resource Profiling:** GPU utilization, memory consumption, and power measurements.
- **Visual Analytics:** Graphs illustrating workload performance and resource utilization.
- **Deployment Documentation:** Cloud execution and reproducibility instructions, with Docker-based GPU containerization being explored.
- **Academic Report:** A technical report documenting the methodology, experimental findings, limitations, and conclusions.

## Deployment and Reproducibility

The GPU workloads have been executed using Google Colab with an NVIDIA Tesla T4 GPU.

Docker-based containerization is the next deployment stage. The objective is to package the required software dependencies and execute CUDA-based workloads inside a container on a compatible NVIDIA GPU system.

Refer to [`docs/cloud-deployment.md`](docs/cloud-deployment.md) for environment setup, workload execution, and deployment documentation.

## Research and References

1. Kirk, D. B., Hwu, W. W., & El Hajj, I. *Programming Massively Parallel Processors: A Hands-on Approach.* Morgan Kaufmann.

2. NVIDIA Corporation. *CUDA C++ Best Practices Guide.*

3. Paszke, A., et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* Advances in Neural Information Processing Systems.

4. Harris, M. (2007). *Optimizing Parallel Reduction in CUDA.* NVIDIA Developer Technology.

5. NVIDIA Corporation. *CUDA C++ Programming Guide.*

6. Numba Development Team. *Numba Documentation: CUDA Support.* https://numba.readthedocs.io/

7. Gonzalez, R. C., & Woods, R. E. *Digital Image Processing.* Pearson.

8. Szeliski, R. *Computer Vision: Algorithms and Applications.* Springer.

---
