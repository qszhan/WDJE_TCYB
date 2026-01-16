# WDJE: Wasserstein Distance Joint Estimation

Python implementation of "To Transfer or Not to Transfer: Unified Transferability Metric and Analysis".

## Overview
WDJE is a unified transferability metric that helps determine whether transfer learning should be performed for both classification and regression tasks under domain and task differences. 
It is built upon two key components: (1) a decision rule that determines whether transfer should be performed by comparing the estimated target risk with and without transfer, yielding the WDJE score, and (2) a computable target risk bound that provides the estimated target transfer risk used inside this decision rule.
Beyond transfer decision-making, WDJE can also be used to select pretrained models, source/target domain pairs.
 
### Key Concept
The WDJE score for determining whether to transfer or not is computed as:
```
WDJE_score = transfer_bound - target_error
```
where the transfer_bound is the upper bound of target risk after transfer learning, and target_error is the true target risk obatined without transfer, based on target data only. 
If WDJE_score <0, the task is transferable as the transfer results in a performance gain. Conversely, the task is nontransferable as the transfer results in a performance loss.



transfer_bound is calculated analytically below:
```
transfer_bound = ε_s + L * λ * W_x + W_y + E_s + L * M * exp(-λ)
```
where:
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
