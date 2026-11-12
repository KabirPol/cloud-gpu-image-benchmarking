# CUDA Fundamentals Using Python

## 1. Introduction

A Graphics Processing Unit (GPU) is designed to perform many computations in parallel. GPUs are useful for workloads that involve large amounts of repetitive mathematical operations, such as image processing, scientific computing and machine learning.

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform. It allows programs to execute computational tasks on NVIDIA GPUs.

In this project, we use **Numba CUDA**, which allows us to write custom GPU kernels using Python.

## 2. CPU vs GPU Computing

### CPU Computing

- A Central Processing Unit (CPU) is designed for general-purpose computing.
- It typically has a smaller number of powerful processing cores.
- It is suitable for sequential operations and general program execution.

### GPU Computing

- A Graphics Processing Unit (GPU) contains many processing cores.
- It can execute numerous operations in parallel.
- It is useful for computationally intensive workloads that can be divided into smaller independent tasks.

## 3. CUDA Programming Using Python

Numba is a Python library that supports GPU programming through CUDA.

The `numba.cuda` module provides functionality to:

- Define GPU kernels.
- Identify GPU devices.
- Configure threads and blocks.
- Transfer data between CPU and GPU memory.
- Execute computations on an NVIDIA GPU.

The `@cuda.jit` decorator is used to compile a Python function as a CUDA kernel.

## 4. Basic CUDA Execution Model

CUDA organizes parallel execution using three main concepts.

### 4.1 Thread

A thread is an individual execution unit responsible for performing a computation.

For example, in vector addition, each thread can calculate one output element.

### 4.2 Block

A block is a group of threads that execute together on the GPU.

Threads within a block can cooperate and share resources.

### 4.3 Grid

A grid is a collection of thread blocks launched for a kernel execution.

The relationship is:

Grid → Blocks → Threads

## 5. Host and Device Memory

CUDA programs involve two types of memory locations.

- **Host memory:** The main system memory used by the CPU.
- **Device memory:** The GPU memory used by GPU computations.

In a typical Numba CUDA workflow:

1. Input data is created in CPU memory.
2. Data is transferred to GPU memory.
3. A CUDA kernel processes the data.
4. Results are copied back to CPU memory.

Data transfer is an important part of GPU computing because it contributes to the overall execution time.

## 6. CUDA Kernel Execution

A CUDA kernel is a function executed by multiple GPU threads.

In Numba CUDA, a kernel can be defined using:

`@cuda.jit`

The function `cuda.grid(1)` calculates the global thread index for a one-dimensional grid.

For vector operations, this index identifies which array element a thread should process.

## 7. Thread and Block Configuration

The number of threads per block determines how many threads are grouped in each block.

The number of blocks required for a one-dimensional array can be calculated as:

**Blocks = ceil(Number of elements / Threads per block)**

For an array of `N` elements and `T` threads per block:

**Blocks = (N + T - 1) // T**

Here, `//` represents integer division, which rounds down the result. Adding `T - 1` ensures that the number of blocks is rounded up when necessary.

**Example:**

For `N = 300` elements and `T = 128` threads per block:

`Blocks = (300 + 128 - 1) // 128`

`Blocks = 427 // 128 = 3`

If the number of elements is not an exact multiple of the threads per block, some threads in the final block may not correspond to valid array elements. Therefore, CUDA kernels should include a boundary check to prevent invalid memory access.

## 8. Vector Addition as a Basic GPU Workload

Vector addition is a simple parallel computing operation.

Given two input vectors:

\[
A=[a_1,a_2,\ldots,a_n]
\]

\[
B=[b_1,b_2,\ldots,b_n]
\]

The output vector is:

\[
C[i]=A[i]+B[i]
\]

Each output element can be calculated independently, making this operation suitable for parallel execution.

In the CUDA implementation, each thread calculates one output element using its global thread index. This allows multiple elements of the output vector to be calculated concurrently.

## 9. Relevance to the Project

Vector addition provides an introductory understanding of GPU programming before implementing the computational workloads selected for this project.

The concepts learned through this experiment will be applied to:

- GPU kernel development.
- Parallel execution.
- GPU memory management.
- CPU and GPU performance comparison.
- Performance analysis of image-processing and machine learning workloads.

## 10. GPU Memory Management and Data Transfers

GPU computing involves transferring data between CPU memory and GPU memory.

### 10.1 Host Memory

Host memory is the main system memory used by the CPU. In Python, NumPy arrays are normally stored in host memory.

### 10.2 Device Memory

Device memory is the memory available on the GPU. CUDA kernels access data stored in device memory during GPU execution.

### 10.3 Host-to-Device Transfer

Host-to-Device (H2D) transfer moves input data from CPU memory to GPU memory.

In Numba CUDA, this can be performed using:

`d_A = cuda.to_device(A)`

Here, `A` is the original NumPy array and `d_A` is its copy in GPU memory.

### 10.4 Device-to-Host Transfer

Device-to-Host (D2H) transfer moves computed results from GPU memory back to CPU memory.

In Numba CUDA, this can be performed using:

`C = d_C.copy_to_host()`

Here, `d_C` is the output array in GPU memory and `C` is the resulting NumPy array in CPU memory.

### 10.5 GPU Memory Allocation

GPU memory must be allocated for data that will be processed by a CUDA kernel.

Numba CUDA provides functions such as:

- `cuda.to_device()` to transfer and allocate data on the GPU.
- `cuda.device_array()` to allocate an array directly in GPU memory.
- `copy_to_host()` to retrieve results from GPU memory.

### 10.6 Importance of Memory Transfers

Data transfers contribute to the overall execution time of a GPU application.

For small workloads, memory-transfer overhead can be significant compared with kernel execution time.

For larger workloads, the computational benefits of GPU parallelism may outweigh the transfer overhead.

Therefore, GPU performance evaluation should consider:

1. Host-to-Device transfer time.
2. GPU kernel execution time.
3. Device-to-Host transfer time.
4. Total end-to-end execution time.

### 10.7 Relevance to Project Workloads

In this project, different computational workloads will be executed on the GPU to study their performance, resource utilization and scalability.

The input data, such as images or training data, will initially be stored in CPU memory and transferred to GPU memory for computation.

The selected workloads include:

- **Sobel edge detection:** Calculates horizontal and vertical image gradients to identify edges.
- **Gaussian blur:** Applies a Gaussian filter to smooth images.
- **Histogram equalization:** Enhances image contrast by redistributing pixel intensity values.
- **Machine learning application:** Uses GPU acceleration for model training and image classification.

Depending on the workload, CUDA kernels or GPU-accelerated machine learning frameworks will perform the computations.

The resulting outputs will be transferred back to CPU memory when required for evaluation and visualization.

Measuring data-transfer time, computation time and overall execution time will help us understand the performance characteristics and overheads of different GPU workloads.
