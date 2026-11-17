# Cloud Deployment and Execution

## 1. Overview

This project uses two separate GPU execution environments to evaluate and validate GPU-based image-processing workloads.

1. **Google Colab with NVIDIA Tesla T4:** Used as the primary environment for performance benchmarking, scalability analysis, optimization experiments and GPU resource monitoring.
2. **Docker with NVIDIA GeForce GTX 1080:** Used as an additional environment for containerized execution, cross-GPU correctness validation and preliminary timing observations.

The two environments are maintained separately because they use different GPUs, software configurations and experimental procedures.

## 2. Primary Cloud Environment: Google Colab

Google Colab was used as the cloud-based GPU computing environment for executing the image-processing and machine learning workloads.

The experiments were performed using an NVIDIA Tesla T4 GPU.

The hardware and software configuration used for the GPU image-processing experiments is shown below:

| Component | Configuration |
|---|---|
| Cloud platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU memory | 15 GB |
| CUDA version | 13.0 |
| NVIDIA driver | 580.82.07 |
| Programming language | Python |
| GPU programming framework | Numba CUDA |
| Array processing | NumPy |
| Data analysis | Pandas |
| Visualization | Matplotlib |

The machine learning workload was implemented separately using PyTorch.

## 3. Environment Setup

A GPU-enabled runtime was selected in Google Colab before executing the experiments.

The Numba CUDA package was installed using:

```python
!pip install "numba-cuda[cu13]"
```

The primary libraries used for the image-processing workloads were:

```python
import numpy as np
import pandas as pd
import math
import time
from numba import cuda, njit
```

Additional libraries, including Matplotlib and scikit-image, were used for visualization and image loading.

For the machine learning workload, PyTorch and Torchvision were used.

## 4. GPU Verification

GPU availability and device information were checked before executing the workloads.

```python
print("CUDA available:", cuda.is_available())

if cuda.is_available():
    print("GPU:", cuda.get_current_device().name)
```

The execution environment successfully detected the NVIDIA Tesla T4 GPU.

GPU driver and memory information were also verified using:

```bash
!nvidia-smi
```

The CUDA-enabled environment was successfully configured for the Numba-based image-processing experiments.

## 5. Workload Execution

The project evaluated three GPU-based image-processing workloads and one machine learning application.

### 5.1 Sobel Edge Detection

The Sobel edge detection workload was implemented using CPU and Numba CUDA GPU approaches.

The experiment included:

- CPU and GPU implementations.
- Correctness verification using a real image.
- Performance evaluation across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- Shared-memory optimization.
- GPU resource utilization monitoring.

### 5.2 Gaussian Blur

The Gaussian blur workload was implemented using a 5 × 5 Gaussian filter.

The experiment included:

- CPU and GPU implementations.
- Real-image correctness verification.
- Performance comparison across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- GPU resource utilization monitoring.

### 5.3 Histogram Equalization

The histogram equalization workload was implemented using CPU and GPU approaches.

The GPU implementation used multiple CUDA kernels for histogram initialization, histogram calculation, cumulative distribution function computation, lookup table generation, and image equalization.

The experiment included:

- CPU and GPU implementations.
- Correctness verification using a real image.
- Performance comparison across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- GPU resource utilization monitoring.

### 5.4 Machine Learning Application

A Convolutional Neural Network (CNN) was implemented using PyTorch and trained on the CIFAR-10 dataset.

The experiment included:

- GPU-based CNN training.
- Training loss monitoring.
- Epoch execution time measurement.
- Model evaluation using the test dataset.

This workload demonstrated the use of a cloud GPU for a machine learning application in addition to CUDA-based image processing.

## 6. Result Collection and Visualization

Execution times, correctness results and GPU resource measurements were collected during the experiments.

The following performance aspects were evaluated:

- CPU execution time.
- GPU kernel execution time.
- GPU end-to-end execution time.
- Kernel and end-to-end speedup.
- Host-to-Device and Device-to-Host transfer time.
- Thread-block configuration.
- GPU utilization.
- GPU memory consumption.
- GPU power consumption.

Performance graphs and visualizations were generated using Matplotlib and saved in the project's `results/figures/` directory.

The experimental notebooks and their outputs are maintained in the project's `notebooks/` directory.

## 7. Docker-Based Deployment and Containerization

### 7.1 Objective

Docker was used to create an isolated GPU execution environment containing the required CUDA toolkit, Python dependencies and Numba CUDA programming framework.

The purpose was to verify that the image-processing workloads could execute inside a container and that the implementations remained correct on a different NVIDIA GPU architecture.

The Docker validation was performed on a Windows 10 lab computer equipped with an NVIDIA GeForce GTX 1080 GPU.

### 7.2 Docker Environment

The following configuration was used:

| Component | Configuration |
|---|---|
| Host operating system | Windows 10 |
| Container platform | Docker Desktop 4.99.0 |
| Docker engine | Linux containers through WSL2 |
| Container operating system | Ubuntu 22.04 |
| GPU | NVIDIA GeForce GTX 1080 |
| GPU memory | 8 GB |
| GPU architecture | Pascal |
| Compute capability | 6.1 |
| NVIDIA driver | 582.53 |
| CUDA toolkit | 12.6 |
| Base Docker image | nvidia/cuda:12.6.3-devel-ubuntu22.04 |
| Python | Python 3.10 |
| GPU programming framework | Numba CUDA |

CUDA 12.6 was selected for the container environment to support the GTX 1080 GPU architecture.

### 7.3 Dockerfile Configuration

A Dockerfile was created using an NVIDIA CUDA development image.

The configuration was:

```dockerfile
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
    "numba-cuda[cu12]" numpy pandas matplotlib scikit-image ipython

WORKDIR /workspace

CMD ["bash"]
```

This configuration installs the Python environment and the libraries required to execute the CUDA-based image-processing workloads.

The CUDA development image provides the CUDA toolkit and compiler tools required for the container environment.

### 7.4 Building the Docker Image

The Docker image was built using:

```cmd
docker build -t gpu-image-benchmark .
```

The image was created successfully using the NVIDIA CUDA 12.6 development base image.

### 7.5 GPU Passthrough Verification

Docker GPU passthrough was verified using:

```cmd
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

The command successfully detected the NVIDIA GeForce GTX 1080 GPU from inside the container.

This confirmed that Docker could access the host NVIDIA GPU through the configured container runtime.

### 7.6 Running the Container

The project directory was mounted inside the container so that the existing workload files could be accessed.

From the project directory, the container was started using:

```cmd
docker run --rm -it --gpus all -v "%cd%:/workspace" gpu-image-benchmark
```

The mounted project directory was available inside the container at `/workspace`.

### 7.7 CUDA and Numba Verification

Inside the running container, CUDA availability was checked using:

```python
from numba import cuda
import numpy as np

print("CUDA available:", cuda.is_available())

if cuda.is_available():
    print("GPU:", cuda.get_current_device().name)

print("NumPy version:", np.__version__)
```

The following checks were successful:

- Numba CUDA detected an available CUDA device.
- The NVIDIA GeForce GTX 1080 was identified.
- CUDA compilation tools were available through NVCC 12.6.85.
- CUDA kernels could be executed using Numba.

### 7.8 Workload Execution and Validation

The existing CUDA implementations were extracted from their respective experimental notebooks and executed inside the Docker container.

The three image-processing workloads were tested:

1. Sobel Edge Detection.
2. Gaussian Blur.
3. Histogram Equalization.

Correctness was verified by comparing CPU and GPU outputs using both real and randomly generated images.

The validation results are summarized below.

| Workload | Validation Input | Resolution | Maximum Absolute Difference | Status |
|---|---|---:|---:|---|
| Sobel Edge Detection | Real Image | 512 × 512 | 0.0 | Passed |
| Sobel Baseline vs Shared Memory | Random Image | 2048 × 2048 | 0.0 | Passed |
| Gaussian Blur | Real Image | 512 × 512 | 0.0 | Passed |
| Gaussian Blur | Random Image | 2048 × 2048 | 0.0 | Passed |
| Histogram Equalization | Real Image | 512 × 512 | 0 | Passed |
| Histogram Equalization | Random Image | 2048 × 2048 | 0 | Passed |

All three workloads successfully passed the correctness checks in the GTX 1080 Docker environment.

### 7.9 Preliminary Timing Observations

Preliminary timing measurements were also collected during the Docker validation.

| Workload | Measurement | Execution Time |
|---|---|---:|
| Sobel Baseline | Average GPU kernel time (30 runs) | 1.0721 ms |
| Sobel Shared Memory | Average GPU kernel time (30 runs) | 1.2317 ms |
| Gaussian Blur | Single GPU end-to-end measurement | 59.0239 ms |
| Histogram Equalization | Single GPU end-to-end measurement | 10.3237 ms |

The Sobel baseline and shared-memory implementations were measured using repeated kernel execution. The shared-memory implementation was approximately 0.87× as fast as the baseline in this preliminary measurement.

The Gaussian blur and histogram equalization values are single end-to-end measurements.

These measurements are retained as preliminary observations and are not used as direct comparisons against the primary Colab T4 benchmark results because the measurement procedures and execution environments differ.

### 7.10 Docker Validation Limitations

The Docker environment was primarily used for cross-GPU correctness validation rather than a complete performance benchmarking campaign.

The following limitations apply:

- The GTX 1080 validation did not include a complete multi-resolution performance study.
- Thread-block configuration and resource-utilization experiments were not repeated comprehensively on the GTX 1080.
- Sobel timing used repeated kernel measurements, whereas Gaussian blur and histogram equalization timings were single end-to-end measurements.
- Preliminary timing values should not be directly compared with the Colab T4 benchmark results.
- GPU utilization, memory and power profiling on the GTX 1080 were not part of this validation.

Therefore, the Colab T4 environment remains the primary source for performance and scalability conclusions.

## 8. Reproducibility

The cloud-based experiments can be reproduced using the corresponding Jupyter notebooks in Google Colab.

### 8.1 Google Colab Execution

The general procedure is:

1. Open the required notebook from the project's GitHub repository.
2. Select a GPU-enabled runtime in Google Colab.
3. Install the required dependencies.
4. Verify GPU availability.
5. Execute the notebook cells in sequence.
6. Collect performance measurements and generated visualizations.

### 8.2 Docker Execution

The Docker-based validation environment can be recreated on a compatible NVIDIA GPU system by following these steps:

1. Install Docker Desktop or an equivalent Docker environment.
2. Configure NVIDIA GPU access through the appropriate container runtime.
3. Ensure that the NVIDIA driver supports the selected CUDA container.
4. Place the Dockerfile in the build directory.
5. Build the Docker image.
6. Start the container with GPU access enabled.
7. Verify CUDA availability using Numba.
8. Execute the required CUDA workloads.

The Docker environment uses the NVIDIA CUDA 12.6 development image and the `numba-cuda[cu12]` package.

Execution times may vary depending on GPU architecture, driver versions, background workloads and system configuration.

## 9. Deployment Limitations

The experiments were conducted using two different GPU environments, each serving a separate purpose.

The following limitations should be considered:

- GPU availability and execution conditions may vary across Colab sessions.
- Performance measurements may be affected by background workloads and cloud resource allocation.
- GPU memory transfer and kernel-launch overhead can influence end-to-end performance.
- The primary performance results are specific to the NVIDIA Tesla T4 environment.
- The GTX 1080 Docker environment was used for additional correctness validation and preliminary timing observations, not a complete second benchmarking study.
- Actual CUDA GPU execution inside Docker requires a compatible NVIDIA GPU, driver and container runtime configuration.

The use of two GPU environments provides additional validation of the workload implementations while keeping the primary performance analysis consistent and reproducible.
