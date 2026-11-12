# Cloud Deployment and Execution

## 1. Cloud Environment

The Sobel edge detection workload was executed using Google Colab as the cloud-based GPU environment.

The following hardware and software configuration was used:

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

## 2. Environment Setup

A GPU-enabled runtime was selected in Google Colab.

The Numba CUDA package was installed using:

```python
!pip install "numba-cuda[cu13]"
```

The required libraries were imported:

```python
import numpy as np
import pandas as pd
import math
import time
from numba import cuda
```

## 3. GPU Verification

GPU availability and device information were checked before executing the workload.

```python
print("CUDA available:", cuda.is_available())

if cuda.is_available():
    print("GPU:", cuda.get_current_device().name)
```

The execution environment successfully detected the NVIDIA Tesla T4 GPU.

## 4. Workload Execution

The Sobel edge detection notebook was uploaded to Google Colab and executed using the GPU runtime.

The execution included:

- CPU and GPU implementations of Sobel edge detection.
- Correctness verification using a real image.
- Performance evaluation across multiple image resolutions.
- GPU memory transfer measurements.
- Thread-block configuration comparison.
- Shared-memory optimization.
- GPU resource monitoring.

## 5. Result Collection

Execution times, correctness results and GPU resource measurements were collected during the experiments.

The generated performance graphs and resource-monitoring data were saved in the project's results directory.

## 6. Reproducibility

The experiment can be reproduced by opening the Sobel notebook in Google Colab, selecting a GPU runtime, installing the required dependencies and executing the notebook cells.

Execution times and resource measurements may vary depending on GPU availability, runtime conditions and hardware configuration.
