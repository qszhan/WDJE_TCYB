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


### Example 1: WDJE for Transfer Decision

Use `WDJE_score` to determine whether transfer learning should be performed. The decision rule is: if `WDJE_score < 0`, transfer is beneficial.

```python
from wdje_open import wdje_score
import numpy as np

# Prepare your data
source_features = np.load('source_features.npy')  # (n_source, d)
source_labels = np.load('source_labels.npy')      # (n_source, num_classes)
target_features = np.load('target_features.npy')  # (n_target, d)
target_labels = np.load('target_labels.npy')      # (n_target, num_classes)

# Errors: source_error from source model, target_error from training on target only
source_error = 0.05   # Error of the pre-trained model on source domain
target_error = 0.30   # Error from training on target data without transfer

# Compute WDJE score
bound_transfer, WDJE_score = wdje_score(
    source_features, source_labels,
    target_features, target_labels,
    num_classes=10,
    source_error=source_error,
    target_error=target_error
)

# Transfer decision
if WDJE_score < 0:
    print(f"Transfer is beneficial (WDJE_score = {WDJE_score:.4f})")
    print("Recommendation: Perform transfer learning")
else:
    print(f"Transfer is NOT beneficial (WDJE_score = {WDJE_score:.4f})")
    print("Recommendation: Train on target data only")
```

### Example 2: WDJE for Model/Domain Selection

Use `bound_transfer` to select the best pre-trained model from a model zoo, or to select the best source/target domain pair. **Select the one with the smallest `bound_transfer`**.

 
```python
from wdje_open import wdje_score
import numpy as np

# Model zoo: different pre-trained models
model_zoo = ['resnet50', 'vit_base', 'efficientnet', 'convnext']

# Target domain data (fixed)
target_features = np.load('target_features.npy')
target_labels = np.load('target_labels.npy')

results = {}
for model_name in model_zoo:
    # Load features extracted by each pre-trained model
    source_features = np.load(f'{model_name}_source_features.npy')
    source_labels = np.load(f'{model_name}_source_labels.npy')
    source_error = np.load(f'{model_name}_source_error.npy')

    bound_transfer, _ = wdje_score(
        source_features, source_labels,
        target_features, target_labels,
        num_classes=10,
        source_error=source_error,
        target_error=0.0  # Not needed for model selection
    )
    results[model_name] = bound_transfer
    print(f"{model_name}: bound_transfer = {bound_transfer:.4f}")

# Select the model with smallest bound_transfer
best_model = min(results, key=results.get)
print(f"\nBest model: {best_model} (bound_transfer = {results[best_model]:.4f})")
```

