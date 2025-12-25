import numpy as np
import pandas as pd
import logging 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from typing import Sequence

def get_best_coefs(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lambda_grid: Sequence[float] | None = None,
    eps: float = 1e-8,
    gamma: float = 1.0,
    ) -> np.ndarray:
    """
    Computes the best coefficients to predict target node y based 
    on its parents X via the adaptive lasso regression (Zou 2006) and BIC score. 
    Whether lambda_grid is provided or not it calculates the lowest BIC score 
    one can obtain for coefficients X out of all lambda values.

    Parameters
    ----------
    X : np.ndarray
        Predictor variables 
    y : np.ndarray
        Target variable 
    * : asterisk operator
        Every argument after the asterisk must be provided with its keyword. All
        arguments after the asterisk are optional.
    lambda_grid : tuple[float], optional
        Lambda values for hyperparameter tuning. The default is None.
    eps : float, optional
         A small value eps to possibly prevent weight calculations from becoming 
         exactly zero as all weights have to be positive. The default is 1e-8.
    gamma : float, optional
        γ tunes how adaptive the second‑stage penalty is: higher γ magnifies 
        the difference between strong and weak pilot coefficients, 
        making strong edges easier to retain and weak edges easier to prune.
        (Zou 2006) uses γ = 1 as it gives the oracle property as sample size 
        grows so default is 1.

    Returns
    -------
    beta_best : np.ndarray
        Returns an array of dimension (j,) where their coefficients 
        predict target node y the best 
 
    """
    
    # 0. decide hyper‑parameters
    if lambda_grid is None:
        lambda0     = 1.0               # pilot penalty
        alpha_desc  = None              # ask sklearn to generate its own grid
    else:
        lambda_grid = np.asarray(lambda_grid, dtype=float)
        lambda0     = float(lambda_grid[0])
        alpha_desc  = np.sort(lambda_grid / 2.0)[::-1]   # high → low


    # 1. pilot Lasso to get weights
    scaler  = StandardScaler(with_mean=True, with_std=True)
    Xs      = scaler.fit_transform(X)
    ys      = y - y.mean()
    beta_sc = Lasso(
        alpha=lambda0 / 2.0,            # factor‑of‑2 caveat
        fit_intercept=False,
        max_iter=10_000,
        tol=1e-4,
        random_state=0,
    ).fit(Xs, ys).coef_

    beta_orig = beta_sc / scaler.scale_
    weights = (np.abs(beta_orig) + eps) ** (-gamma)

    # 2. weighted Lasso path (implemented via column scaling)
    Xweighted = X / weights[np.newaxis, :]

    alphas, coefs, _ = Lasso.path(
        Xweighted,
        ys,
        alphas=alpha_desc,      # None ⇒ sklearn chooses
    )

    # 3. pick best λ via BIC
    rss = ((ys[:, None] - Xweighted @ coefs) ** 2).sum(axis=0)
    n, _ = X.shape
    df   = (coefs != 0).sum(axis=0)
    bic  = n * np.log(rss / n) + df * np.log(n)
    j_best = bic.argmin()
    
    beta_best = (coefs[:, j_best]) / weights
    return beta_best

    
def adaptive_lasso_algorithm(data: pd.DataFrame,
                             *,
                             lambda_grid: Sequence[float] | None = None,
                             order: tuple = None,
                             eps: float = 1e-8,
                             gamma: float = 1.0,
                             verbose: bool = False) -> pd.DataFrame:
    """
    Computes coefficients of the adaptive lasso regression (Zou, 2006) for 
    each target node j and parent nodes 1...j-1 given some topological order
    and returns adjacency matrix A representing the final DAG. 
    (Ali and George 2009, Algorithm 1 Penalized Likelihood Estimation of DAGs)

    Parameters
    ----------
    data : pd.DataFrame
        A dataframe with columns representing each node X1, ..., Xj
        
    * : Asterisk operator
        Every argument after the asterisk must be provided with its keyword. All
        arguments after the asterisk are optional.
        
    lambda_grid : list[float], optional
        A list of lambda values for hyperparameter tuning. 
        If lamda_grid is not provided then the lasso function 
        from sklearn creates its own grid. The default is None.
        
    order : tuple, optional
        A tuple representing the topological order of the nodes. 
        If order argument not provided then the order of the columns is assumed to 
        follow the same topological order. The default is None.
        
    eps : float, optional
        A small value eps to possibly prevent weight calculations from becoming 
        exactly zero as all weights have to be positive. The default is 1e-8.
        
    gamma : float, optional
        γ tunes how adaptive the second‑stage penalty is: higher γ magnifies 
        the difference between strong and weak pilot coefficients, 
        making strong edges easier to retain and weak edges easier to prune.
        (Zou 2006) uses γ = 1 as it gives the oracle property as sample size 
        grows so default is 1.
        
    Raises
    ------
    ValueError
        ValueError is raised if the order argument provided by the user does 
        not cover all the columns of the data.
        
    Returns
    -------
    adj_df : pd.DataFrame
        Returns an strictly lower triangular adjacency matrix (a DAG)
        in the form of a pandas dataframe. Where element is 1 then 
        it represents a directed edge Xi -> Xk from 
        column i and row k and no edge if zero. 
        
    """
    
    # logging option
    if verbose:
       logging.basicConfig(
           filename="adaptivelasso.log",  
           filemode="w",                   
           level=logging.INFO,
           format="%(asctime)s | %(levelname)s | %(message)s",
           datefmt="%H:%M:%S",
       )
    
    # Reorder columns of data if order provided
    if order is None:
        order = tuple(data.columns)
    else:
        if missing := set(data.columns) - set(order):
            raise ValueError(f"order argument has missing columns: {missing!r}")
            
    data = data.reindex(columns=order)
    
    # Initialization 
    X = data.to_numpy()
    _, p = data.shape
    A = np.zeros((p,p), dtype=np.int8)
    
    # Compute Weighted Lasso coefficients for each target node j and predictor nodes 1..j-1
    for j in range(1,p):
        target_node = X[:, j]
        parent_nodes = X[:, :j]
        beta = get_best_coefs(parent_nodes, 
                              target_node, 
                              lambda_grid=lambda_grid, 
                              eps=eps, 
                              gamma=gamma)
        A[j, :j] = beta != 0
        
        if verbose:
            logging.info(
                f"{order[j]} ~ {[order[i] for i, b in enumerate(beta) if b != 0]} \
                Coefficients: {[float(np.round(b,3)) for b in beta if b != 0]}"
            )
            
    
    # Final Adjacency Matrix DAG
    adj_df = pd.DataFrame(A, index=order, columns=order)
    return adj_df

# ------------------------ Example ----------------------------------------

if __name__ == "__main__":
    np.random.seed(123)
    n = 100
    x1 = np.random.normal(0, 1, n)                       # root
    x2 = np.random.normal(0, 1, n)                       # root
    x3 = 0.7 * x1 + np.random.normal(0, 0.3, n)          # x1 →
    x4 = -0.5 * x1 + 0.6 * x2 + np.random.normal(0, 0.3, n)  # x1,x2 →
    x5 = 0.8 * x3 + np.random.normal(0, 0.3, n)          # x3 →
    x6 = 0.5 * x4 + 0.4 * x2 + np.random.normal(0, 0.3, n)   # x4,x2 →
    x7 = 0.6 * x5 - 0.3 * x1 + np.random.normal(0, 0.3, n)   # x5,x1 →
    x8 = 0.5 * x6 + np.random.normal(0, 0.3, n)          # x6 →
    x9 = 0.4 * x7 + 0.3 * x3 + np.random.normal(0, 0.3, n)   # x7,x3 →
    x10 = 0.5 * x8 + 0.4 * x4 + np.random.normal(0, 0.3, n)  # x8,x4 →

    df = pd.DataFrame(
        np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10]),
        columns=[f"x{i}" for i in range(1, 11)]
    )
    adj_df = adaptive_lasso_algorithm(df, lambda_grid=np.logspace(-2,1,20), verbose=True)
    edges  = [(p, c) for c in adj_df.index
                        for p in adj_df.columns
                        if adj_df.loc[c, p] == 1]  
    print(f"Adjacency matrix: \n {adj_df}")
    print(f"Directed edges for DAG: \n {edges}")
    
    df = adj_df.to_numpy()
    rows, cols = np.triu_indices(len(df), k=0)
    zeroes_diag_or_above = df[rows, cols]
    assert np.count_nonzero(zeroes_diag_or_above) == 0, "Adjacency matrix should be strictly lower diagonal for a DAG"
    








    
    
    
    
    
    
    
    
    
    
    
    
    
    
    