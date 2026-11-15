# Cloud Deployment and Execution

## 1. Cloud Environment

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

## 2. Environment Setup

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

## 3. GPU Verification

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

## 4. Workload Execution

The project evaluated three GPU-based image-processing workloads and one machine learning application.

### 4.1 Sobel Edge Detection

The Sobel edge detection workload was implemented using CPU and Numba CUDA GPU approaches.

The experiment included:

- CPU and GPU implementations.
- Correctness verification using a real image.
- Performance evaluation across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- Shared-memory optimization.
- GPU resource utilization monitoring.

### 4.2 Gaussian Blur

The Gaussian blur workload was implemented using a 5 × 5 Gaussian filter.

The experiment included:

- CPU and GPU implementations.
- Real-image correctness verification.
- Performance comparison across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- GPU resource utilization monitoring.

### 4.3 Histogram Equalization

The histogram equalization workload was implemented using CPU and GPU approaches.

The GPU implementation used multiple CUDA kernels for histogram initialization, histogram calculation, cumulative distribution function computation, lookup table generation, and image equalization.

The experiment included:

- CPU and GPU implementations.
- Correctness verification using a real image.
- Performance comparison across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- GPU resource utilization monitoring.

### 4.4 Machine Learning Application

A Convolutional Neural Network (CNN) was implemented using PyTorch and trained on the CIFAR-10 dataset.

The experiment included:

- GPU-based CNN training.
- Training loss monitoring.
- Epoch execution time measurement.
- Model evaluation using the test dataset.

This workload demonstrated the use of a cloud GPU for a machine learning application in addition to CUDA-based image processing.

## 5. Result Collection and Visualization

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

## 6. Docker-Based Deployment and Containerization

Docker-based containerization is planned to provide a reproducible execution environment for GPU workloads.

The objective is to package the required Python dependencies, CUDA-compatible libraries and workload code into a Docker image.

The planned deployment procedure includes:

1. Preparing a Dockerfile and dependency configuration.
2. Building a Docker image with the required Python and CUDA-compatible environment.
3. Configuring NVIDIA GPU access using NVIDIA Container Toolkit.
4. Running the container on a compatible NVIDIA GPU system.
5. Verifying GPU availability from inside the container.
6. Executing an existing CUDA-based image-processing workload inside the container.

The Docker configuration is being prepared on an Apple silicon MacBook Air. Actual NVIDIA CUDA execution and GPU container testing will be performed on a compatible lab computer equipped with an NVIDIA GPU.

**Note:** Docker installation and GPU container execution are separate stages. Successful Docker installation alone does not establish CUDA GPU access inside a container.

## 7. Reproducibility

The cloud-based experiments can be reproduced using the corresponding Jupyter notebooks in Google Colab.

The general procedure is:

1. Open the required notebook from the project's GitHub repository.
2. Select a GPU-enabled runtime in Google Colab.
3. Install the required dependencies.
4. Verify GPU availability.
5. Execute the notebook cells in sequence.
6. Collect performance measurements and generated visualizations.

The Docker-based execution procedure will be documented separately after successful GPU container testing.

Execution times and resource measurements may vary depending on GPU availability, hardware configuration, background workloads and cloud runtime conditions.

## 8. Deployment Limitations

The experiments were conducted using Google Colab, where GPU availability and resource allocation may vary.

The following limitations should be considered:

- GPU availability and execution conditions may vary across Colab sessions.
- Performance measurements may be affected by background workloads and cloud resource allocation.
- GPU memory transfer and kernel-launch overhead can influence end-to-end performance.
- The measured performance is specific to the NVIDIA Tesla T4 environment.
- Actual CUDA GPU execution inside Docker requires a compatible NVIDIA GPU, driver and container runtime configuration.

