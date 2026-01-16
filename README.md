# WDJE: Wasserstein Distance Joint Estimation

Python implementation of "To Transfer or Not to Transfer: Unified Transferability Metric and Analysis".

## Overview

WDJE (Wasserstein Distance Joint Estimation) provides a theoretical framework for estimating the transfer learning bound between two domains. It uses optimal transport theory to measure the distribution discrepancy in both feature space and label space.

### Key Concept

The transfer bound is computed as:

```
Bound = ε_s + L * λ * W_x + W_y + E_s + L * M * exp(-λ)
```

Where:
- `ε_s`: Source domain error
- `L`: Lipschitz constant of the hypothesis
- `λ`: Trade-off parameter balancing feature vs. label distance
- `W_x`: Wasserstein distance between source and target features
- `W_y`: Wasserstein distance between source and target labels
- `E_s`: Expected norm of unlabeled source predictions
- `M`: Scaling constant (typically log(num_classes))

## Installation

### Dependencies

```bash
pip install numpy scipy pot scikit-learn
```

- `numpy`: Numerical computations
- `scipy`: Distance computations (cdist)
- `pot`: Python Optimal Transport library
- `scikit-learn`: Data preprocessing (MinMaxScaler)

## Usage

### Basic Usage

```python
from wdje_open import wdje_score
import numpy as np

# Prepare your data
source_features = np.load('source_features.npy')  # (n_source, d)
source_labels = np.load('source_labels.npy')      # (n_source, num_classes)
target_features = np.load('target_features.npy')  # (n_target, d)
target_labels = np.load('target_labels.npy')      # (n_target, num_classes)

# Compute WDJE score
result = wdje_score(
    source_features=source_features,
    source_labels=source_labels,
    target_features=target_features,
    target_labels=target_labels,
    num_classes=10
)

print(f"Transfer Bound: {result['bound_custom']:.4f}")
```
