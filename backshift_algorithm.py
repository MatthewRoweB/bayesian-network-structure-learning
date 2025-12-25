## BACKSHIFT Algorithm

import numpy as np
import pandas as pd
from scipy.linalg import eigh
import pathlib, logging
from logging.handlers import RotatingFileHandler

# ------ Paths -----------------------------------------------------
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
LOG_FILE = PROJECT_ROOT / "logs" / "backshift.log"
LOG_FILE.parent.mkdir(exist_ok=True, parents=True)

# ----- Logging ----------------------------------------------------
handler = RotatingFileHandler(LOG_FILE, maxBytes=5_242_880, backupCount=3)
formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO, handlers=[handler])

logger = logging.getLogger(__name__)
logger.info("Rotating File Folder Initialized")

# ----- BACKSHIFT_Algorithm ----------------------------------------

def mean_diff_cov_matrices(deltas: list[np.ndarray]) -> np.ndarray:
    """
    Compute eigen-vectors of the mean ΔΣ matrix.

    Parameters
    ----------
    deltas : list of np.ndarray (each p×p)
        Covariance-difference matrices Σ^(j) − Σ^baseline.

    Returns
    -------
    np.ndarray  (p×p)
        Columns are the eigen-vectors (joint diagonaliser).
    """
    mean_delta = sum(deltas) / len(deltas)
    eigenvalues, eigenvectors = eigh(mean_delta)

    rad = np.abs(eigenvalues).max()
    logger.info("Eigen-vals ρ=%.4f  %s", rad, np.round(eigenvalues, 4).tolist())
    if rad >= 1:
        logger.warning("Spectral radius ≥ 1 — B̂ may be unstable.")
    return eigenvectors

def backshift_algorithm(
    df_long: pd.DataFrame,
    *,
    baseline_env,
    env_col: str = "env",
    threshold: float = 1e-3,
    ) -> pd.DataFrame:
    """
    Original BACKSHIFT (eigen-basis) returning a pandas DataFrame.

    Parameters
    ----------
    df_long      : long-form DataFrame (env column + numeric vars)
    baseline_env : label inside `env_col` that marks the baseline rows
    env_col      : str, default 'env'
    threshold    : float, default 1e-3 — entries |B_ij|<τ set to 0

    Returns
    -------
    pandas.DataFrame
        Connectivity matrix B̂ (rows & cols = variable names).
    """
    # ① select variables (all columns except env)
    var_cols = df_long.columns.difference([env_col])
    p        = len(var_cols)

    # ② get covariances per environment
    g   = df_long.groupby(env_col)
    Σ0  = g.get_group(baseline_env)[var_cols].cov().values

    # ③ ΔΣ list (list-comprehension, not generator)
    deltas = [
        g.get_group(e)[var_cols].cov().values - Σ0
        for e in g.groups if e != baseline_env
    ]
    if not deltas:
        raise ValueError("Need at least one non-baseline environment")

    # ④ eigen-basis & connectivity matrix
    M      = mean_diff_cov_matrices(deltas)
    B_hat  = np.eye(p) - np.linalg.inv(M)
    B_hat[np.abs(B_hat) < threshold] = 0     # sparsify

    return pd.DataFrame(B_hat, index=var_cols, columns=var_cols)


# Example
rng     = np.random.default_rng(42)
vars_   = ["Temp", "Humidity", "Fertilizer", "Rainfall", "Yield"]
n_rows  = 6

def base_df():
    mu = np.array([20, 50, 10, 5, 200])
    sigma = np.diag([4, 9, 1, 1, 36])
    return pd.DataFrame(rng.multivariate_normal(mu, sigma, size=n_rows),
                        columns=vars_)

baseline = base_df().assign(env=0)

shifts = {
    1: {"Temp": +3},
    2: {"Humidity": +12},
    3: {"Fertilizer": +4},
    4: {"Temp": -2, "Rainfall": +3},
    5: {"Temp": +2, "Humidity": -5, "Fertilizer": +3},
}
def shift_df(eid, delta):
    df = baseline.drop(columns="env").copy()
    for col, d in delta.items():
        df[col] += d
    return df.assign(env=eid)

df_list = [baseline] + [shift_df(e, d) for e, d in shifts.items()]

# --- concat into long-form (env first) ------------------------------------
df_long = pd.concat(df_list, ignore_index=True, join="inner")

# --- run BACKSHIFT --------------------------------------------------------
B = backshift_algorithm(df_long, baseline_env=0, threshold=1e-3)
print("\nConnectivity matrix B̂:")
print(B.round(4))






















