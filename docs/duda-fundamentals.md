
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

## 9. Relevance to the Project

Vector addition provides an introductory understanding of GPU programming before implementing the selected image-processing workload: Sobel edge detection.

The concepts learned through this experiment will be applied to:

- GPU kernel development.
- Parallel execution.
- GPU memory management.
- CPU and GPU performance comparison.
- Performance analysis of image-processing workloads.

## 10. References

1. Numba Documentation: https://numba.readthedocs.io/
2. Numba CUDA Documentation: https://numba.readthedocs.io/en/stable/cuda/
3. NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
