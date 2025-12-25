from sklearn.linear_model import Lasso
from functools import lru_cache
from typing import Literal
import pandas as pd

# --------------------------- Meinshausen Algorithm --------------------------

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized z-score each column:
      X_ij ← (X_ij – mean_j) / std_j
    """
    return (df - df.mean()) / df.std(ddof=1)

@lru_cache(maxsize=None)
def nodewise_lasso(col: str,
                   std_df: pd.DataFrame,
                   lam: float,
                   *,
                   max_iter: int,
                   tol: float
                  ) -> set[str]:
    """
    Fit a Lasso of the column `col` (endogenous) on all the other columns
    in `std_df` (exogenous). Return the set of predictor‐column names
    whose coefficients are non‐zero.
    """
    # 1) Define endogenous (target) and exogenous (predictors)
    target_var     = std_df[col]
    predictors_var = std_df.drop(columns=col)         

    # 2) Fit Lasso in scikit-learn
    model = Lasso(alpha=lam, max_iter=max_iter, tol=tol).fit(predictors_var, target_var)

    # 3) Boolean mask of non-zero coefficients (vectorized)
    mask = model.coef_ != 0.0

    # 4) Select only those predictor names where mask is True
    neighbours_col = set(predictors_var.columns[mask])

    return neighbours_col

def fit_all_nodewise_lasso(df: pd.DataFrame,
                           lam: float,
                           *,
                           max_iter: int,
                           tol: float
                          ) -> dict[str, set[str]]:
    """
    Run nodewise_lasso for every column in df.
    Returns a dict mapping each column name to its set of neighbours.
    """
    # apply takes each column Series (col), calls nodewise_lasso on it
    neighbours_series = df.apply(
        lambda col: nodewise_lasso(
            col.name,    # the column label
            df,
            lam,
            max_iter=max_iter,
            tol=tol
        ),
        axis=0
    )
    # convert the resulting Series (col_name -> set) to a plain dict
    return neighbours_series.to_dict()

def select_edges_by_rule(adj: dict[int, set[int]], 
                         rule: Literal["or", "and", "both"] = "or"):
    """
    From directed adj → undirected edges under 'or', 'and', or 'both'.

    Parameters
    ----------
    adj  : dict mapping node -> set(directed neighbours)
    rule : "or", "and", or "both"

    Returns
    -------
    - If rule in {"or","and"}: set of edges {(i,j),...}
    - If rule == "both": dict{"or": set, "and": set}
    """
    # 1) Validate rule
    if rule not in ("or", "and", "both"):
        raise ValueError(f"rule must be 'or', 'and', or 'both', not {rule!r}")

    # 2) Build the two direction‐checking predicates
    or_pred  = lambda i, j: (j in adj[i] or i in adj[j])
    and_pred = lambda i, j: (j in adj[i] and i in adj[j])

    # 3) One extractor for ANY predicate
    extract = lambda pred: {
        frozenset((i, j))
        for i, nbrs in adj.items()
        for j in nbrs
        if i < j and pred(i, j)
    }

    # 4) Dispatch
    if rule == "or":
        return extract(or_pred)

    elif rule == "and":
        return extract(and_pred)

    else:  # rule == "both"
        return {
            "or":  extract(or_pred),
            "and": extract(and_pred),
        }

def meinshausen_algorithm(data: pd.DataFrame,
                          lam: float,
                          rule: Literal["or", "and", "both"] = "or",
                          *,
                          max_iter: int = 1000,
                          tol: float    = 1e-3) -> set[frozenset[str, str]]:
    """
    Neighbourhood-selection via nodewise Lasso.

    Parameters
    ----------
    data     : pd.DataFrame, shape (n, p)
    lam      : float           Lasso penalty (alpha)
    rule     : "or", "and", or "both"
    max_iter : int             max CD sweeps per Lasso
    tol      : float           solver convergence tolerance

    Returns
    -------
    - If rule in {"or","and"}: set of edges {(i,j),...}
    - If rule=="both": dict{"or": set, "and": set}
    """
    
    std_data = standardize(data)
    adj = fit_all_nodewise_lasso(std_data, lam, max_iter=max_iter, tol=tol)
    graph_skeleton = select_edges_by_rule(adj, rule)
    
    return graph_skeleton





















