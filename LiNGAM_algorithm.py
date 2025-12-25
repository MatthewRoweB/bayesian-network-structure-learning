import pandas as pd 
import numpy as np
from typing import Literal
from sklearn.decomposition import FastICA
from sklearn.utils import resample
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
import statsmodels.api as sm
from scipy.stats import chi2
from lingam.utils import evaluate_model_fit
import heapq
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



# Algorithm B 
def permutation_for_b(B: NDArray, tol: float):
    
    n = B.shape[1]
    remaining = np.arange(n)
    order = []
    
    while remaining.size:
        # rows whose absolute values are all <= tol  (i.e., all zeros)
        zero_rows_local = np.where(np.all(np.abs(B) <= tol, axis=1))[0]
        if zero_rows_local.size == 0:
            return False  # cannot find a zero row ⇒ not DAG

        # pick one zero row (deterministic: the smallest index)
        i_local = int(zero_rows_local[0])
        i_global = int(remaining[i_local])
        order.append(i_global)

        # remove i-th row and column from the working matrix and the index map
        B = np.delete(np.delete(B, i_local, axis=0), i_local, axis=1)
        remaining = np.delete(remaining, i_local)

    # If we got here, we found a full topological order of original indices
    row_order = order
    col_order = order
    return row_order, col_order
    

# Algorithm C
def causal_order_dag_test(B: NDArray, tol: float):
    
    m = B.shape[0]

    # --- Step 1: zero k = m(m+1)/2 smallest entries (absolute value) ---
    k = m * (m + 1) // 2
    idx = np.argpartition(np.abs(B), k - 1, axis=None)[:k]  # flat indices of k smallest
    B.ravel()[idx] = 0.0

    # --- Step 2+: loop until Algorithm B succeeds ---
    while True:
        perm = permutation_for_b(B, tol)  # -> (row_order, col_order) or False
        if perm is not False:
            row_order, col_order = perm
            B_line = B[np.ix_(row_order, col_order)]
            return B_line

        # Zero the next smallest *non-zero* entry and continue
        absB = np.abs(B).ravel()
        absB[absB == 0.0] = np.inf           # skip places already zero
        nxt = np.argmin(absB)                # flat index of current smallest non-zero
        
        B.ravel()[nxt] = 0.0
        
    return B 

def bootstrap_pvals(X, B, bootstrap_wald_stats:int=500, alpha=float, random_state=None):
    """
    Compute bootstrap SEs for nonzero b_ij in strictly lower-triangular B.
    Returns dict of (i,j) -> (coef, SE, Wald, pval).
    """
    n, p = X.shape
    X = X.to_numpy()

    for i in range(p):
        parents = np.where(B[i, :] != 0)[0]
        if parents.size == 0:
            continue
        # Original fit
        model = sm.OLS(X[:, i], sm.add_constant(X[:, parents])).fit()
        for k, j in enumerate(parents):
            coef = model.params[k+1]   # skip constant
            # Bootstrap reps
            boots = []
            for r in range(bootstrap_wald_stats):
                Xb = resample(X, replace=True, n_samples=n, random_state=random_state)
                modb = sm.OLS(Xb[:, i], sm.add_constant(Xb[:, parents])).fit()
                boots.append(modb.params[k+1])
            se = np.std(boots, ddof=1)
            wald = (coef/se)**2 if se > 0 else np.inf
            p_val = chi2.sf(wald, 1)
    yield ([i,j], p_val)                # Collection of edges [i,j] with their p_val from wald statistic


def model_fit_and_difference(B_before, B_after, X, alpha):
    # get chi2, DoF, p for both models
    full = evaluate_model_fit(B_before, X).iloc[0]
    red  = evaluate_model_fit(B_after, X).iloc[0]
    evaluate_model_fit(B_after,  X)
    chi2_full, df_full, p_full = full['chi2'], full.get('dof', full.get('DoF')), full.get('chi2 p-value')   # error 
    chi2_red,  df_red,  p_red  = red['chi2'],  red.get('dof',  red.get('DoF')),  red.get('chi2 p-value')

    # Δχ² test
    delta_chi2, delta_df = chi2_full - chi2_red, df_full - df_red
    p_diff = 1 - chi2.cdf(delta_chi2, delta_df)

    # accept prune if reduced model still fits AND difference is non-sig
    return (p_red > alpha) and (p_diff > alpha)



def prune_B_line(X: pd.DataFrame, B_line: NDArray, bootstrap_wald_stats: int, alpha: float, random_state):
    
    non_significant_edges = [(-p_val, edge) for edge, p_val in bootstrap_pvals(X, B_line, 
                                                    bootstrap_wald_stats, alpha, random_state)]
    heapq.heapify(non_significant_edges)
    B_full = B_line.copy()
    
    while non_significant_edges:
        
        _, edge = heapq.heappop(non_significant_edges)      # Pop least significant edge        
        B_constrained = B_full.copy()
        i_c, j_c = map(int, edge)
        B_constrained[i_c, j_c] = 0
    
        if model_fit_and_difference(B_full, B_constrained, X, alpha=alpha):
            B_full = B_constrained
            
    B_line_pruned = B_full
    return B_line_pruned
    

def lingam_algorithm(data: pd.DataFrame, 
                     *,
                     fun: Literal['logcosh', 'exp', 'cube'] | callable = 'logcosh',
                     prune_edges: bool = False, 
                     alpha: float = 0.05, 
                     tol: float = 1e-12,
                     max_iter: int = 200,
                     whiten_solver = 'eigh',
                     algorithm='parallel',
                     boostrap_wald_stats: int = 500,
                     random_state: float | None = None):
    
    
    
    centered = data.sub(data.mean(axis=1), axis=0)
    
    ica = FastICA(
    n_components=data.shape[1],  # p
    algorithm=algorithm,
    random_state=random_state,
    whiten='unit-variance',
    max_iter=max_iter,
    fun=fun,
    tol=tol,
    whiten_solver=whiten_solver
    )
    
    S = ica.fit_transform(data)
    W = ica.components_             # (p, p)  <-- square unmixing
    
    C = 1.0 / (np.abs(W) + 1e-12)
    
    row_ind, col_ind = linear_sum_assignment(C)
    row_order = np.empty_like(col_ind)
    
    row_order[col_ind] = row_ind
    
    W_perm = W[row_order, :]                              # Permutation of W for largest diagonals
    
    W_perm = W_perm / np.diag(W_perm)[:, np.newaxis]
    
    B = np.eye(data.shape[1]) - W_perm
    np.fill_diagonal(B, 0)                                # Diagonals 0 check


    B_line = causal_order_dag_test(B, tol)
    
    if prune_edges:
        B_line_pruned = prune_B_line(centered, B_line, boostrap_wald_stats, alpha, random_state)
        return B_line_pruned
    
    return B_line


if __name__ == "__main__": 
    
    np.random.seed(42)  # for reproducibility
    
    n = 1000   # samples
    p = 30     # variables
    
    # 1️⃣ Create a random strictly lower-triangular weight matrix B (DAG)
    B = np.tril(np.random.uniform(-0.8, 0.8, size=(p, p)), k=-1)
    
    # 2️⃣ Generate independent, non-Gaussian noise terms (Laplace is fine)
    E = np.random.laplace(loc=0, scale=1, size=(n, p))
    
    # 3️⃣ Generate variables linearly according to the DAG order
    X = np.zeros((n, p))
    for i in range(p):
        X[:, i] = X @ B[:, i] + E[:, i]
    
    # 4️⃣ Wrap into a pandas DataFrame
    columns = [f"X{i+1}" for i in range(p)]
    df = pd.DataFrame(X, columns=columns)
    
    print(df.head())
    print("\nShape:", df.shape)
    
    result = lingam_algorithm(df, max_iter=2000, tol=0.01, fun='logcosh', prune_edges=True)
    print(pd.DataFrame(result))
    
    if np.allclose(result, np.tril(result, k=-1)):
        print("B is strictly lower triangular")
    else:
        raise ValueError("B is not strictly lower triangular")


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










