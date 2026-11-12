# Problem Formulation

## 1. Project Title

**Performance, Resource Utilization, and Scalability Analysis of Cloud-Based GPU Workloads for Image Processing**

## 2. Background and Motivation

Image processing is widely used in computer vision, medical imaging, remote sensing, satellite imagery, and other computational applications. Processing large images or batches of images can require substantial computation.

Central Processing Units (CPUs) are designed for general-purpose computing, whereas Graphics Processing Units (GPUs) can execute many suitable operations in parallel. Cloud computing provides access to GPU resources without requiring users to own dedicated GPU hardware.

However, GPU acceleration does not guarantee better performance in every situation. Execution time can depend on image size, batch size, data-transfer overhead, GPU resources, and implementation choices. Cloud-based GPU execution can also involve resource constraints and additional execution overhead.

Therefore, experimental evaluation is needed to understand how different image-processing workloads perform on CPUs and cloud GPUs under different workload conditions.

## 3. Problem Statement

The problem addressed in this project is to investigate the performance, resource utilization, and scalability of selected image-processing workloads executed using cloud-based GPU resources.

The study compares CPU and GPU execution under defined experimental conditions and examines how changes in workload size affect execution time, throughput, and resource requirements.

The primary workload is Sobel edge detection, with Gaussian blur and histogram equalization included as additional image-processing workloads. The experiments aim to identify the conditions under which cloud GPU execution provides performance benefits and the limitations that may affect those benefits.

## 4. Research Question

**How do workload characteristics, particularly image resolution and batch size, affect the execution time, throughput, and resource utilization of image-processing workloads when comparing CPU execution with cloud GPU execution?**

## 5. Project Objectives

The objectives of this project are:

1. To implement Sobel edge detection as the primary image-processing workload, with Gaussian blur and histogram equalization as additional workloads.
2. To develop CPU and GPU implementations of the selected image-processing operations.
3. To measure execution time across different image resolutions and, where feasible, batch sizes.
4. To compare CPU and cloud GPU performance using equivalent workload conditions.
5. To investigate GPU thread-block configurations and evaluate kernel optimization techniques for the Sobel workload.
6. To measure GPU utilization, memory consumption, and power usage where the available environment permits.
7. To analyze the effects of workload size and computational configurations on performance and resource requirements.
8. To document the implementation and experimental environment to support reproducibility.

## 6. Scope of the Project

The project focuses on image-processing workloads executed on a CPU and an NVIDIA GPU available through a cloud computing platform.

The selected workloads are:

- **Sobel edge detection:** The primary workload, used for detailed CPU–GPU performance comparison, thread-block configuration experiments, and kernel optimization evaluation.
- **Gaussian blur:** An additional image-filtering workload used to evaluate GPU acceleration for convolution-based image processing.
- **Histogram equalization:** An additional workload used to evaluate GPU performance for histogram-based image processing.

The workloads are implemented using Python, NumPy, and CUDA through Numba. Google Colab with an NVIDIA Tesla T4 GPU is used as the cloud-based execution environment.

The experiments include CPU–GPU performance comparison across different image resolutions. For the primary Sobel workload, different thread-block configurations are evaluated, and shared-memory optimization is investigated.

Performance is analyzed using kernel execution time and end-to-end processing time. GPU utilization, memory consumption, and power usage are monitored where supported by the cloud environment.

Scalability is examined by studying how performance changes with workload size and, where feasible, batch size. The project also considers cloud deployment and containerization as part of the practical evaluation of cloud-based GPU workloads.

The project focuses on experimental evaluation and performance analysis rather than developing a new image-processing algorithm. Image resizing is not included as an experimental workload.

## 7. Performance Metrics

The following metrics are used or considered for evaluation:

- **Execution time:** Time required to process the workload.
- **GPU kernel execution time:** Time taken by the GPU kernel to perform the computation, excluding host-to-device and device-to-host transfers.
- **End-to-end processing time:** Total time required for processing, including relevant memory allocation, data transfers, GPU execution, and result retrieval.
- **Throughput:** Number of images processed per unit of time.
- **Speedup:** Ratio of CPU execution time to GPU execution time for equivalent work.
- **GPU utilization:** Percentage of GPU processing capacity reported as active during execution.
- **GPU memory usage:** Memory consumed during workload execution.
- **GPU power consumption:** Power usage reported by the GPU monitoring tools during execution.
- **Data-transfer overhead:** Time associated with transferring data between host and GPU memory, where measurable.
- **Scalability:** How performance changes as workload size or batch size increases.

Measurements will distinguish kernel execution time from end-to-end processing time wherever possible. Resource measurements will be interpreted according to the capabilities and sampling limitations of the cloud environment.

## 8. Expected Outcomes

The expected outcomes are:

1. CPU and GPU implementations of the selected image-processing workloads.
2. Experimental measurements covering different image resolutions and workload configurations.
3. A quantitative comparison of CPU and cloud GPU execution.
4. An evaluation of thread-block configurations and kernel optimization techniques for Sobel edge detection.
5. Measurements and analysis of GPU utilization, memory consumption, and power usage.
6. Graphs and tables illustrating performance, resource-usage, and scalability trends.
7. An analysis of the conditions and limitations affecting GPU acceleration in a cloud environment.
8. A reproducible project repository and a final conference-style report.

These are intended outcomes; actual performance conclusions will be determined by the experiments.
