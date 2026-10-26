# Literature Review

## 1. Introduction

Graphics Processing Units (GPUs) are widely used to accelerate computationally intensive workloads because they can execute many operations in parallel. Image-processing applications, such as filtering, image transformation, and computer vision, can benefit from this parallel processing capability. However, the performance improvement depends on the workload, data movement between CPU and GPU memory, and the computing resources available.

Cloud computing provides access to GPU-enabled infrastructure without requiring users to own dedicated GPU hardware. However, evaluating a cloud-based GPU application requires more than measuring its computation time. Resource utilization, data-transfer overhead, deployment configuration, and scalability can also influence its overall performance.

This literature review examines existing work on GPU-accelerated computing, image-processing performance, and cloud deployment. The objective is to identify suitable evaluation methods and establish the basis for the proposed experiments.

## 2. Review of Related Work

### 2.1 GPU-Accelerated Cloud Computing

The study *GPU-accelerated cloud computing services and performance evaluation*, published in Simulation Modelling Practice and Theory in 2025, investigates GPU-accelerated computing as an alternative to traditional CPU-based cloud computing. The authors benchmark GPU computing against multithreaded CPU computing across several application areas, including machine learning and large-scale image processing. The study also considers an autoscaling strategy designed to improve GPU resource utilization before adding further GPUs.

This work is directly relevant to the present project because it considers both computational performance and resource allocation in cloud environments. It provides a useful basis for investigating CPU–GPU performance differences and understanding the role of resource utilization. The detailed experimental setup and reported results must be examined to determine which methods can be adapted to the present study.

### 2.2 Cloud Deployment and Scalability for Computer Vision

Zheng et al. (2026), in *Scaling Computer Vision: A Comparative Analysis of Cloud Infrastructures for AI-Based Image Processing and Classification Applications*, compare different AWS deployment architectures for a real-world image-processing and classification application. The evaluated architectures include a single EC2 instance, a deployment using multiple EC2 instances, and a container-based deployment using Amazon ECS.

The authors evaluate execution time, resource utilization, scalability, and cost. Their reported results indicate that the multiple-EC2 deployment performed best among the evaluated configurations, while the container-based deployment offered management advantages with additional overhead.

This study demonstrates that application performance depends not only on the image-processing workload but also on how the application is deployed and its resources are organized. However, its focus on deployment architectures differs from a controlled comparison of individual image-processing operations across CPU and GPU execution. This distinction motivates further investigation of workload-level performance under different input sizes and batch sizes.

### 2.3 GPU Performance Measurement

NVIDIA's CUDA C++ Best Practices Guide provides technical guidance on evaluating and optimizing CUDA applications. It discusses performance measurement, memory operations, and data transfers between host and device memory.

This guidance is important for designing reliable CPU–GPU experiments. Measuring only GPU kernel execution may not represent the total time required to process an image, particularly when input data must first be transferred to GPU memory and results must subsequently be returned to the CPU. Therefore, the present project should distinguish kernel execution time from end-to-end processing time whenever the experimental setup permits.

## 3. Synthesis of the Reviewed Work

The reviewed sources address complementary aspects of GPU computing. GPU benchmarking helps explain computational performance, while cloud deployment studies demonstrate how infrastructure configuration influences scalability, resource utilization, and cost. Technical documentation provides practical guidance for conducting and interpreting performance measurements.

Together, these sources suggest that GPU performance should be evaluated using multiple metrics rather than execution time alone. The evaluation should consider input workload characteristics, CPU and GPU computation, data-transfer overhead, and the resources used during execution.

## 4. Relevance to the Proposed Project

Building on these studies, the proposed project will investigate image-processing workloads under CPU and cloud-GPU execution. Candidate operations include grayscale conversion, Gaussian blur, and Sobel edge detection, subject to implementation and resource availability.

The experiments will examine how image resolution and batch size affect execution time and throughput. GPU utilization and memory usage will also be measured where the selected cloud environment provides suitable monitoring tools. The analysis will distinguish computation time from end-to-end processing time where possible.

The final research gap and contribution will be refined after reviewing additional studies and comparing their experimental methods with the proposed setup.

## References

1. *GPU-accelerated cloud computing services and performance evaluation*. Simulation Modelling Practice and Theory, 144 (2025), Article 103181. https://doi.org/10.1016/j.simpat.2025.103181

2. Zheng, H., Reaño, C., Castillo, A., Ariño-Sales, J. F., Igual, Á., and Igual, C. (2026). *Scaling Computer Vision: A Comparative Analysis of Cloud Infrastructures for AI-Based Image Processing and Classification Applications*. Electronics, 15(9), 1953. https://doi.org/10.3390/electronics15091953

3. NVIDIA. *CUDA C++ Best Practices Guide*. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
