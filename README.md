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

## 📊 Expected Deliverables
The final submission will follow a structured journal/conference format containing:
1. Complete Python/PyTorch benchmarking codebase hosted on GitHub.
2. Comprehensive hardware profiling data tables (Execution Time, VRAM allocation, Peak GPU Usage).
3. Scalability visualization graphs illustrating data throughput trends.
