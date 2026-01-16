 
import numpy as np
import ot
from scipy.spatial.distance import cdist
from typing import Tuple


def compute_wasserstein_distance(
    source: np.ndarray,
    target: np.ndarray,
    p: int = 2
) -> Tuple[float, float]:
    """
    Compute the Wasserstein distance between source and target distributions.

    This function uses Earth Mover's Distance (EMD) from optimal transport theory
    to measure the distance between two distributions.

    Args:
        source: Source distribution features/labels, shape (n_source, d)
        target: Target distribution features/labels, shape (n_target, d)
        p: Order of the Wasserstein distance (1 or 2). Default is 2.

    Returns:
        Wasserstein distance  
    

    Example:
        >>> source_features = np.random.randn(100, 512)
        >>> target_features = np.random.randn(50, 512)
        >>> w_dist = compute_wasserstein_distance(source_features, target_features)
    """
    
    if p == 2:
        cost_matrix = cdist(source, target, metric='sqeuclidean')
    else:
        cost_matrix = cdist(source, target, metric='euclidean')

  
    source_weights = ot.unif(source.shape[0])
    target_weights = ot.unif(target.shape[0]) 
 
    dis_wasserstein = ot.emd2(source_weights, target_weights, cost_matrix)

    return dis_wasserstein


def estimate_transfer_bound(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    lipschitz_constant: float,
    source_error: float = 0.0,
    M: float = np.log(100),
    lmbda: float = 0.5,
    p: int = 2
) -> dict:
    """
    Estimate the target bound after transfer using Wasserstein distances.

    This implements the WDJE bound estimation:
        bound = source_error + L * lambda * W_x + W_y + E_s + constant

    where:
        - source_error: source domain error
        - W_x: Wasserstein distance between source and target features
        - W_y: Wasserstein distance between source and target labels
        - L: Lipschitz constant
        - E_s: Expected norm of source predictions
        - constant: L * M * exp(-lambda)

    Args:
        source_features: Source domain features, shape (n_source, d)
        source_labels: Source domain labels/predictions, shape (n_source, c) or (n_source,)
        target_features: Target domain features, shape (n_target, d)
        target_labels: Target domain labels (labeled portion), shape (n_labeled, c) or (n_labeled,)
        lipschitz_constant: Lipschitz constant of the hypothesis
        source_error: Source domain error (default: 0.0)
        M: loss upper bound constant (M=logC for classification, M=1 for regression)
        lambda: Trade-off parameter (default: 0.5, detailed analysis for choosing lambda is in Section S3 the supplementary material)
        p: Order of Wasserstein distance (default: 2)

    Returns:
        Dictionary containing:
            - 'bound_custom': Transfer bound using custom Wasserstein computation
            - 'bound_emd2': Transfer bound using EMD2 Wasserstein
            - 'wasserstein_x': Feature space Wasserstein distance (custom)
            - 'wasserstein_y': Label space Wasserstein distance (custom)
            - 'wasserstein_x_emd2': Feature space Wasserstein distance (EMD2)
            - 'wasserstein_y_emd2': Label space Wasserstein distance (EMD2)
            - 'expected_norm': Expected norm of unlabeled predictions
            - 'regularization_constant': L * M * exp(-lambda)
            - 'lambda': Trade-off parameter used

    Example:
        >>> src_feat = np.random.randn(100, 512)
        >>> src_labels = np.random.randn(100, 10)
        >>> tar_feat = np.random.randn(50, 512)
        >>> tar_labels = np.random.randn(30, 10)  # 30 labeled samples
        >>> lipz = 0.1
        >>> result = estimate_transfer_bound(src_feat, src_labels, tar_feat, tar_labels, lipz)
        >>> print(f"Transfer bound: {result['bound_custom']:.4f}")
    """
 

    num_target_labeled = len(target_labels)
    source_labels_labeled = source_labels[:num_target_labeled]
    source_labels_unlabeled = source_labels[num_target_labeled:]
    w_x = compute_wasserstein_distance(source_features, target_features, p)
    w_y = 0.0 if num_target_labeled == 0 else compute_wasserstein_distance(source_labels_labeled, target_labels, p)
    if len(source_labels_unlabeled) > 0:
        row_norms = np.linalg.norm(source_labels_unlabeled, axis=1)
        expected_norm = np.mean(row_norms)           
    else:
        expected_norm = 0.0
    reg_constant = lipschitz_constant * M * np.exp(-lmbda)
    # Compute transfer bounds
    bound = source_error + lipschitz_constant * lmbda * w_x + w_y + expected_norm + reg_constant
    
    return {
        'bound_transfer': bound,
        'wasserstein_x': w_x,
        'wasserstein_y': w_y,
        'expected_norm': expected_norm,
        'regularization_constant': reg_constant,
        'lambda': lmbda
    }


def normalize_features(features: np.ndarray, method: str = 'l2') -> np.ndarray:
    """
    Normalize features using specified method.

    Args:
        features: Input features, shape (n_samples, d)
        method: Normalization method ('l2' or 'minmax')

    Returns:
        Normalized features with same shape as input
    """
    if method == 'l2':
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1, norms)
        return features / norms
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        return scaler.fit_transform(features)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_lipschitz_constant(
    features: np.ndarray,
    num_classes: int,
    num_samples: int
) -> float:
    """
    Compute the Lipschitz constant for clasification.
    The Lipschitz constant is estimated as:
        L = ||features||_2 * (num_classes - 1) / (num_classes * num_samples)
    """
    feature_norm = np.linalg.norm(features, ord=2)
    constant = (num_classes - 1) / (num_classes * num_samples)
    return feature_norm * constant


# Convenience function for end-to-end WDJE computation
def wdje_score(
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
    target_labels: np.ndarray,
    num_classes: int,
    source_error: float = 0.0,
    target_error: float = 0.0,
    lmbda: float = 0.5,
    normalize: bool = True
) -> dict:
    """
    Compute the WDJE (Wasserstein Distance Joint Estimation) transferability score.

    This is a convenience function that combines all steps:
    1. Feature normalization
    2. Lipschitz constant estimation
    3. Transfer bound computation

    Args:
        source_features: Source domain features
        source_labels: Source domain labels/predictions
        target_features: Target domain features
        target_labels: Target domain labels (can be partial)
        num_classes: Number of classes in the task
        source_error: Source domain error  
        target_error: Target domain error obtained from the few labelled samples without transfer  
        lmbda: Trade-off parameter (default: 0.5)
        normalize: Whether to L2-normalize features (default: True)

    Returns:
        Dictionary with transfer bounds and intermediate values
 
    """
    # Normalize features
    if normalize:
        source_features = normalize_features(source_features, method='l2')
        target_features = normalize_features(target_features, method='l2')

    # Compute Lipschitz constant
    lipschitz = compute_lipschitz_constant(
        target_features,
        num_classes,
        len(target_features)
    )

    # Compute transfer bound
    result = estimate_transfer_bound(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_labels=target_labels,
        lipschitz_constant=lipschitz,
        source_error=source_error,
        lmbda=lmbda
    )
    bound_transfer = result['bound_transfer'] 
    WJDE_score = bound_transfer - target_error
    return bound_transfer, WJDE_score


if __name__ == "__main__":
 

    # Generate synthetic data
    n_source, n_target = 100, 50
    feature_dim = 512
    num_classes = 10
    source_error = 0.4    
    target_error = 1.1
    source_features = np.random.randn(n_source, feature_dim)
    source_labels = np.random.randn(n_source, num_classes)
    target_features = np.random.randn(n_target, feature_dim)
    target_labels = np.random.randn(n_target, num_classes)

    # Compute WDJE score
    bound_transfer, WJDE_score = wdje_score(
        source_features, source_labels,
        target_features, target_labels,
        num_classes,
        source_error, target_error
    )
    if WJDE_score < 0:
        print("The target domain is transferable.")
    else:
        print("The target domain is not transferable.")

 
