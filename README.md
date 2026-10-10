# Performance, Resource Utilization, and Scalability Analysis of Cloud-Based GPU Image Workloads

## 📌 Project Details
- **Topic:** Topic 23: Performance, resource utilization, and scalability analysis of cloud-based GPU workloads
- **Workload:** High-Throughput Image Processing / Dataset Augmentation
- **Environment:** Google Colab (NVIDIA T4 GPU)

## 📝 Project Overview
This project evaluates the computational efficiency, resource utilization, and scalability of executing large-scale batch image augmentation pipelines on cloud-based GPU infrastructure. By running intensive, pixel-wise image transformations (such as resizing, geometric flipping, and Gaussian blurring) across varying dataset sizes, this study profiles how effectively a cloud GPU utilizes its streaming multiprocessors and VRAM under heavy, parallelized computer vision workloads.

## 🚀 Key Objectives
1. **Pipeline Implementation:** Build a modular batch image preprocessing script using Python and PyTorch (Torchvision) to handle high-throughput transformations.
2. **Resource Profiling:** Monitor and log GPU memory (VRAM) consumption, core utilization, and system overhead across different batch sizes (e.g., 100, 1,000, and 10,000 images).
3. **Scalability Analysis:** Evaluate processing throughput (images processed per second) to identify hardware bottlenecks and performance saturation limits.

## 🛠️ Expected Tools & Resources
- **Language:** Python
- **Core Libraries:** PyTorch (Torchvision), NumPy, Matplotlib
- **Infrastructure:** Google Colab (utilizing CPU vs. NVIDIA T4 Cloud GPUs)

## 📊 Project Deliverables
- **Codebase:** A modular, well-documented Python/PyTorch benchmarking script hosted entirely in this GitHub repository.
- **Experimental Data:** Structured hardware profiling datasets capturing precise execution times, VRAM allocation spikes, and peak resource utilization metrics.
- **Visual Analytics:** Performance and scalability line graphs plotting image processing throughput against varying batch sizes using Matplotlib.
- **Academic Report:** A comprehensive, publication-style technical paper detailing the project's methodology, empirical findings, and architectural conclusions.

## 📖 Research & References
*The following foundational literature and resources guide the benchmarking methodologies for this study:*

1. **Kirk, D. B., Hwu, W. W., & El Hajj, I.** *Programming Massively Parallel Processors: A Hands-on Approach.* Morgan Kaufmann. (Chapters 1–2: Introduction to throughput-oriented GPU architectures and thread execution models).
2. **NVIDIA Corporation.** *CUDA C++ Best Practices Guide.* (Sections on memory coalescing and optimization strategies for parallel streaming multiprocessors).
3. **Paszke, A., et al.** *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* (Foundational reference for orchestrating CUDA-accelerated batch operations without low-level C++ overhead).
