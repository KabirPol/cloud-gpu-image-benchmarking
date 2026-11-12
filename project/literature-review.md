# Literature Review

## 1. Introduction

Graphics Processing Units (GPUs) are widely used to accelerate computationally intensive workloads because they can execute many operations in parallel. Unlike CPUs, which are designed for a broad range of general-purpose tasks, GPUs are particularly effective for workloads involving large numbers of similar computations. Image-processing applications, such as filtering, blurring, histogram equalization, and edge detection, can benefit from this parallel processing capability.

Cloud computing provides access to GPU-enabled infrastructure without requiring users to own dedicated GPU hardware. However, evaluating a cloud-based GPU application requires more than measuring its computation time. Data-transfer overhead, memory consumption, GPU utilization, deployment configuration, and scalability can also influence overall performance.

This literature review examines existing research on GPU computing, CUDA-based image processing, Python GPU programming, cloud deployment, and GPU performance evaluation. The objective is to identify suitable evaluation methods and establish the foundation for the proposed experiments.

## 2. Review of Related Work

### 2.1 GPU Computing and Parallel Processing

Owens et al. (2008), in *GPU Computing*, discuss the development of GPUs as programmable parallel processors and their application to general-purpose computing. The paper reviews GPU architecture, programming models, tools, and computational applications. It explains how GPU parallelism can provide significant performance improvements for computationally demanding problems.

This work provides the theoretical background for understanding GPU acceleration and the differences between CPU and GPU execution. It establishes the importance of identifying workloads that can benefit from parallel processing.

### 2.2 GPU-Accelerated Cloud Computing

Collins, De Luca, and Chen (2025), in *GPU-accelerated cloud computing services and performance evaluation*, investigate GPU-accelerated computing as an alternative to traditional CPU-based cloud computing. The authors benchmark GPU computing against multithreaded CPU computing across several application areas, including machine learning and large-scale image processing.

The study also introduces an autoscaling approach designed to maximize the utilization of an available GPU before allocating additional GPUs. It discusses performance optimization strategies involving memory management, concurrency, and workload distribution.

This work is directly relevant to the present project because it considers both computational performance and resource allocation in cloud environments. It provides a basis for investigating CPU–GPU performance differences and understanding the importance of efficient GPU resource utilization.

### 2.3 CPU and GPU Performance Comparison in Image Processing

Hangün and Eyecioğlu (2017), in *Performance Comparison Between OpenCV Built in CPU and GPU Functions on Image Processing Operations*, investigate the performance of image-processing operations using OpenCV's built-in CPU and CUDA GPU functions.

The study considers operations including resizing, thresholding, histogram equalization, and edge detection. It compares execution times across different image sizes using CPU and NVIDIA GPU implementations. The edge-detection operation uses the Canny algorithm.

The study is relevant to the present project because it demonstrates the use of image-processing workloads to evaluate CPU and GPU performance and highlights the importance of input image size when comparing execution times.

However, the present project differs in its implementation approach. It uses Python, NumPy, and Numba CUDA to implement image-processing operations, with Sobel edge detection as the primary workload rather than relying on OpenCV's built-in CUDA functions or Canny edge detection. The project also extends the comparison by examining GPU resource utilization, memory consumption, power usage, and cloud-based execution.

### 2.4 CUDA-Based Image Processing

Saha et al. (2016), in *Implementation of Image Enhancement Algorithms and Recursive Ray Tracing using CUDA*, implement several image-processing algorithms using NVIDIA CUDA. The study includes point-processing operations such as brightening, darkening, negative filtering, and RGB-to-grayscale conversion. It also evaluates convolution-based operations, including Sobel edge detection, low-pass filtering, and high-pass filtering.

The authors compare CPU and GPU implementations and investigate performance using different thread-block configurations. Their experiments demonstrate the application of CUDA parallelism to image-processing operations.

This study is particularly relevant to the present project because it includes Sobel edge detection and examines the influence of thread-block configurations on GPU performance. It provides a useful reference for implementing and evaluating the Sobel workload using CUDA.

### 2.5 Python-Based GPU Computing and Data Transfer

Xu et al. (2022), in *Influence of Data Amount, Data Type and Implementation Packages in GPU Coding*, investigate GPU programming in Python by comparing the Numba and CuPy packages. The study examines the influence of data amount and data type on GPU execution performance. It also analyzes data-transfer time between GPU and CPU memory and its effect on total execution time.

The authors report that data amount, data type, and implementation package can influence GPU performance. Their results also indicate that using single-precision data types can improve computation time in Numba implementations.

This work is particularly relevant to the present project because it uses Numba for Python-based GPU computing. It supports the consideration of data types, workload size, and CPU–GPU data-transfer overhead when designing performance experiments.

### 2.6 Cloud Deployment and Scalability for Computer Vision

Zheng et al. (2026), in *Scaling Computer Vision: A Comparative Analysis of Cloud Infrastructures for AI-Based Image Processing and Classification Applications*, compare different AWS deployment architectures for a real-world image-processing and classification application.

The evaluated architectures include a single EC2 instance, a deployment using multiple EC2 instances, and a container-based deployment using Amazon ECS. The authors evaluate execution time, resource utilization, scalability, and cost. Their results highlight the performance and resource-consumption differences between the evaluated deployment configurations.

This study demonstrates that application performance depends not only on the image-processing workload but also on how the application is deployed and its computational resources are organized. It is relevant to the deployment, containerization, and scalability aspects of the present project.

### 2.7 Cloud Benchmarking and Performance Analysis

Dancheva, Alonso, and Barton (2024), in *Cloud Benchmarking and Performance Analysis of an HPC Application in Amazon EC2*, investigate the performance of a high-performance computing application deployed on Amazon EC2 cloud infrastructure.

The study evaluates cloud-based execution and analyzes the performance of the application using benchmarking experiments. It discusses the use of cloud infrastructure for computationally intensive workloads and provides an example of evaluating cloud resources for high-performance computing.

Although the application differs from the image-processing workloads considered in the present project, the study is relevant to the evaluation of computational performance in cloud environments. It provides a reference for designing experiments that assess execution performance using cloud-based resources.

### 2.8 GPU Performance Monitoring in Virtualized Environments

Belkhiri and Dagenais (2024), in *Analyzing GPU Performance in Virtualized Environments: A Case Study*, investigate GPU performance analysis in virtualized environments. The authors develop a performance-monitoring approach using software tracing and analyze GPU-related information from virtualized systems.

The study focuses on collecting and organizing performance data to understand GPU behavior and identify performance-related issues. It also discusses the importance of monitoring GPU resources in virtualized environments.

This work is relevant to the present project because cloud GPU execution involves virtualized computing resources. It provides supporting research for examining GPU utilization and resource behavior during workload execution.

### 2.9 GPU Performance Measurement and Optimization

NVIDIA's *CUDA C++ Best Practices Guide* provides technical guidance on evaluating and optimizing CUDA applications. It discusses performance measurement, memory operations, and data transfers between host and device memory.

This guidance is important for designing reliable CPU–GPU experiments. Measuring only GPU kernel execution may not represent the total time required to process an image, particularly when input data must first be transferred to GPU memory and results must subsequently be returned to the CPU.

Therefore, the present project distinguishes GPU kernel execution time from end-to-end processing time. This helps evaluate both the computational performance of the GPU and the overhead associated with data transfers and other processing steps.

## 3. Synthesis of the Reviewed Work

The reviewed studies address complementary aspects of GPU computing and cloud-based workload evaluation.

Research on GPU computing and CUDA-based image processing establishes the benefits of parallel execution for computationally intensive operations. Studies comparing CPU and GPU implementations demonstrate the importance of evaluating different image sizes and thread-block configurations.

Research on Python GPU programming highlights the influence of data types, workload sizes, and memory transfers on overall performance. Cloud benchmarking and deployment studies demonstrate that infrastructure configuration, resource allocation, and virtualization can influence application performance and scalability.

Together, these studies indicate that GPU performance should be evaluated using multiple metrics rather than execution time alone. The evaluation should consider workload characteristics, CPU and GPU computation, data-transfer overhead, resource utilization, and the cloud environment in which the application is executed.

## 4. Relevance to the Proposed Project

Building on the reviewed studies, the proposed project investigates the performance, resource utilization, and scalability of image-processing workloads in a cloud-based GPU environment.

Sobel edge detection is selected as the primary workload. Gaussian blur and histogram equalization are included as additional image-processing workloads to examine GPU performance across different types of operations.

The workloads are implemented using Python, NumPy, and CUDA through Numba. Google Colab with an NVIDIA Tesla T4 GPU is used as the cloud-based execution environment.

The experiments compare CPU and GPU execution across different image resolutions. Performance is evaluated using execution time, kernel speedup, and end-to-end speedup. Different GPU thread-block configurations are also examined for the Sobel workload.

GPU utilization, memory consumption, and power usage are monitored to understand resource requirements during execution. The project further considers cloud deployment, containerization, and scalability as part of evaluating the practical use of cloud-based GPU workloads.

## 5. Research Gap

Existing studies demonstrate the benefits of GPU acceleration for image processing and examine the influence of programming approaches, cloud infrastructure, and deployment configurations on application performance.

However, the reviewed studies address these aspects separately or use different implementation environments and workloads. The present project focuses on custom Python-based CUDA implementations of image-processing operations in a cloud GPU environment.

It combines CPU–GPU performance comparison, evaluation across different image resolutions, kernel and end-to-end execution time, GPU resource monitoring, and workload scalability within a single experimental study.

The project therefore aims to provide an experimental evaluation of the performance and resource requirements of image-processing workloads executed using cloud-based GPU resources.

## References

1. Owens, J. D., Houston, M., Luebke, D., Green, S., Stone, J. E., & Phillips, J. C. (2008). GPU computing. *Proceedings of the IEEE, 96*(5), 879–899. https://doi.org/10.1109/JPROC.2008.917757

2. Collins, Z., De Luca, G., & Chen, Y. (2025). GPU-accelerated cloud computing services and performance evaluation. *Simulation Modelling Practice and Theory, 144*, 103181. https://doi.org/10.1016/j.simpat.2025.103181

3. Hangün, B., & Eyecioğlu, Ö. (2017). Performance comparison between OpenCV built in CPU and GPU functions on image processing operations. *International Journal of Engineering Science and Application, 1*(2), 34–41.

4. Saha, D., Darji, K., Patel, N., & Thakore, D. (2016). Implementation of image enhancement algorithms and recursive ray tracing using CUDA. *Procedia Computer Science, 79*, 516–524. https://doi.org/10.1016/j.procs.2016.03.066

5. Xu, P., Sun, M.-Y., Gao, Y.-J., Du, T.-J., Hu, J.-M., & Zhang, J.-J. (2022). Influence of data amount, data type and implementation packages in GPU coding. *Array, 16*, 100261. https://doi.org/10.1016/j.array.2022.100261

6. Zheng, H., Reaño, C., Castillo, A., Ariño-Sales, J. F., Igual, Á., & Igual, C. (2026). Scaling computer vision: A comparative analysis of cloud infrastructures for AI-based image processing and classification applications. *Electronics, 15*(9), 1953. https://doi.org/10.3390/electronics15091953

7. Dancheva, T., Alonso, U., & Barton, M. (2024). Cloud benchmarking and performance analysis of an HPC application in Amazon EC2. *Cluster Computing, 27*, 2273–2290. https://doi.org/10.1007/s10586-023-04060-4

8. Belkhiri, A., & Dagenais, M. (2024). Analyzing GPU performance in virtualized environments: A case study. *Future Internet, 16*(3), 72. https://doi.org/10.3390/fi16030072

9. NVIDIA. *CUDA C++ Best Practices Guide*. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
