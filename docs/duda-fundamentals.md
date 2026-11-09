# Introduction to CUDA Programming

## 1. What is CUDA?

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows programmers to use NVIDIA GPUs for general-purpose computing.

Applications include:
- Image processing
- Machine learning
- Deep learning
- Scientific simulations
- Matrix computations

## 2. CPU vs GPU

### CPU
- Contains relatively few powerful cores.
- Optimized for sequential and general-purpose tasks.
- Handles complex control operations.

### GPU
- Contains many processing cores.
- Designed for parallel execution.
- Suitable for processing large amounts of similar data.

## 3. CUDA Architecture

- Host: CPU and its memory.
- Device: GPU and its memory.

Typical execution sequence:

1. Prepare data on the CPU.
2. Transfer data to GPU memory.
3. Launch a GPU kernel.
4. Execute the kernel using GPU threads.
5. Transfer results back to the CPU.

## 4. CUDA Execution Hierarchy

Grid -> Blocks -> Threads

### Thread
The smallest execution unit in the CUDA programming model.

### Block
A group of threads that can cooperate and share memory.

### Grid
A collection of blocks launched for a kernel execution.

## 5. Important CUDA Terminology

- Kernel: Function executed on the GPU.
- Host: CPU side.
- Device: GPU side.
- Global memory: Large GPU memory accessible by threads.
- Shared memory: Fast memory shared by threads within a block.
- Thread ID: Identifies a thread within its block.
- Block ID: Identifies a block within its grid.

## 6. Applications of CUDA in This Project

CUDA will be used to accelerate image-processing operations, particularly Sobel edge detection.

The project will compare CPU and GPU implementations using execution time, throughput, speedup and resource utilization.
