import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Literal
from numpy.typing import NDArray



def slope_estimator(driver: np.ndarray, targets: np.ndarray, *,
                    allow_duplicates: bool = False) -> np.ndarray:
    """
    Compute slope-based directional scores for IGCI.

    Parameters
    ----------
    driver : (n,)
        Sorted base variable (e.g., Xi_sorted).
    targets : (n, p)
        All variables aligned to the same ordering as 'driver'.
    allow_duplicates : bool, default=False
        - False → raise an error if Δx = 0 (strict IGCI assumption).
        - True  → safely ignore Δx = 0 pairs by setting their contribution to 0.

    Returns
    -------
    slopes : (p,)
        Mean log slope per target.
    """
    # consecutive differences
    dx = np.diff(driver)                     # shape (n-1, 1)
    dy = np.diff(targets, axis=0)            # shape (n-1, p)

    # strict mode: enforce continuous, strictly monotone assumption
    if not allow_duplicates and np.any(dx == 0):
        raise ValueError(
            "Duplicate values detected in driver variable. "
            "Set allow_duplicates=True to ignore them (log(0)=0 convention)."
        )

    # compute ratio safely (avoid division by zero)
    if allow_duplicates:
        ratio = np.divide(
            np.abs(dy),
            dx,
            out=np.zeros_like(dy, dtype=float),
            where=(dx != 0)
        )
    else:
        ratio = np.abs(dy / dx)

    # apply log(0)=0 rule
    log_ratio = np.where(ratio > 0, np.log(ratio), 0.0)

    # average across all samples (axis 0)
    slopes = np.mean(log_ratio, axis=0)
    return slopes

def entropy_estimator(X_sorted, Y_sorted):

    pass


def information_geometric_causal_inference(data: pd.DataFrame, 
                                           reference: Literal['uniform', 'gaussian'] = 'uniform',
                                           estimator: Literal['slope', 'entropy'] = 'slope',
                                           sorting_stability: Literal['quicksort', 'mergesort', 'stable'] = 'quicksort',
                                           nearly_equal_threshold: None | float = None
                                           ) -> list[tuple[str, str, float]]:
    
    reference_map = {'uniform': MinMaxScaler, 
                      'gaussian': StandardScaler}
    estimator_map = {'slope' : slope_estimator,
                     'entropy': entropy_estimator}
    
    X = data.to_numpy(dtype=float, copy=False)
    X = reference_map[reference]().fit_transform(X)
    n, p = X.shape
    
    edge_list: list[tuple[str, str, float]] = []
    
    for i in range(p):
        # 1) Sort the whole dataset by column i (driver Xi) — per-iteration argsort
        idx = np.argsort(X[:, i], kind=sorting_stability)[:, np.newaxis]   # (n,1)
        X_sorted = np.take_along_axis(X, idx, axis=0)                      # (n,p)
        Xi_sorted = X_sorted[:, i][:, np.newaxis]                          # (n,)
    
        # 2) Directional scores (vectorized over all targets j)
        cx = slope_estimator(Xi_sorted, X_sorted)   # Xi -> Xj, shape (p,)
        cy = slope_estimator(X_sorted, Xi_sorted)   # Xj -> Xi, shape (p,)
        diff = cx - cy                               # shape (p,)
    
        # 3) Indices by sign (no [0] indexing; diagonal assumed exactly 0)
        pos_idx, = np.where(diff > 0)               # Xi -> Xj winners
        neg_idx, = np.where(diff < 0)               # Xj -> Xi winners
    
        # 4) Coefficients per sign (use cx for positives, cy for negatives)
        #    (You asked to avoid diff for weights)
        edges_pos = ((f"x{i+1}", f"x{j+1}", float(w)) for j, w in zip(pos_idx, cx[pos_idx]))
        edges_neg = ((f"x{j+1}", f"x{i+1}", float(w)) for j, w in zip(neg_idx, cy[neg_idx]))
    
        # 5) Append lazily (no inner j-loop)
        edge_list.extend(edges_pos)
        edge_list.extend(edges_neg)

    return edge_list
        
    
np.random.seed(0)

x = np.random.uniform(0, 1, 1000)
y = np.sin(3 * x) + 0.05 * np.random.randn(1000)
z = np.cos(3.8 * x) + 0.05 * np.random.randn(1000)
p = np.sin(2 * z) + 0.05 * np.random.laplace(loc=0, scale=1, size=1000)

df = pd.DataFrame({"X": x,
                   "Y": y,
                   "Z": z,
                   "P": p})    
    
result = information_geometric_causal_inference(df)
print(result)    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    









