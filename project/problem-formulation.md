# Problem Formulation

## 1. Project Title

**Performance, Resource Utilization, and Scalability Analysis of Cloud-Based GPU Workloads for Image Processing**

## 2. Background and Motivation

Image processing is widely used in computer vision, medical imaging, remote sensing, satellite imagery, and other computational applications. Processing large images or batches of images can require substantial computation.

Central Processing Units (CPUs) are designed for general-purpose computing, whereas Graphics Processing Units (GPUs) can execute many suitable operations in parallel. Cloud computing provides access to GPU resources without requiring users to own dedicated GPU hardware.

However, GPU acceleration does not guarantee better performance in every situation. Execution time can depend on image size, batch size, data-transfer overhead, GPU resources, and implementation choices. Cloud-based GPU execution can also involve resource constraints and differences in execution overhead.

Therefore, experimental evaluation is needed to understand how image-processing workloads perform on CPUs and cloud GPUs under different workload conditions.

## 3. Problem Statement

The problem addressed in this project is to investigate the performance, resource utilization, and scalability of selected image-processing workloads executed using cloud-based GPU resources.

The study will compare CPU and GPU execution under defined experimental conditions and examine how changing the workload size affects execution time, throughput, and resource requirements.

The aim is to use measured experimental results to identify the conditions under which cloud GPU execution provides performance benefits and the limitations that may affect those benefits.

## 4. Research Question

**How do image resolution and batch size affect the execution time, throughput, and resource utilization of selected image-processing workloads when comparing CPU execution with cloud GPU execution?**

## 5. Project Objectives

The objectives of this project are:

1. To select suitable image-processing operations for experimental evaluation.
2. To implement equivalent CPU and GPU versions of the selected operations.
3. To measure execution time under different image resolutions and batch sizes.
4. To compare CPU and cloud GPU performance using consistent workload conditions.
5. To measure GPU utilization and memory usage where the available environment permits.
6. To investigate how performance changes as workload size increases.
7. To analyze the experimental results using tables, graphs, and relevant performance metrics.
8. To document the implementation and experimental environment to support reproducibility.

## 6. Scope of the Project

The project will focus on image-processing workloads that can be executed on a CPU and an NVIDIA GPU available through a cloud computing platform.

Candidate operations include:

- Grayscale conversion
- Gaussian blur
- Sobel edge detection

The final operations and implementation framework will be selected after checking software compatibility and GPU availability.

The experiments will vary image resolution and batch size while recording execution time and throughput. Additional measurements may include GPU utilization, GPU memory usage, data-transfer overhead, and scalability across available computational configurations.

The project will focus on experimental evaluation and performance analysis rather than developing a new image-processing algorithm.

## 7. Performance Metrics

The following metrics are proposed for evaluation:

- **Execution time:** Time required to process the workload.
- **Throughput:** Number of images processed per unit of time.
- **Speedup:** Ratio of CPU execution time to GPU execution time for equivalent work.
- **GPU utilization:** Percentage of GPU processing capacity reported as active during execution, where measurable.
- **GPU memory usage:** Memory consumed during the workload, where measurable.
- **Data-transfer overhead:** Time associated with transferring data between host and GPU memory, where measurable.
- **Scalability:** How performance changes as workload size or available computational resources change.

Measurements will distinguish kernel execution time from end-to-end processing time when possible.

## 8. Expected Outcomes

The expected outcomes are:

1. A working CPU baseline and GPU implementation for the selected image-processing operations.
2. A set of experimental measurements covering different workload sizes.
3. A quantitative comparison of CPU and cloud GPU execution.
4. Graphs and tables illustrating performance and resource-usage trends.
5. An analysis of the conditions and limitations affecting GPU acceleration and scalability.
6. A reproducible project repository and a final conference-style report.

These are intended outcomes; the actual performance conclusions will be determined by the experiments.
