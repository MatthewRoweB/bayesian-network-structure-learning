from itertools import combinations
import heapq
from typing import Literal
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler
from pgmpy.estimators.CITests import pearsonr
import numpy as np

#-------------------------------Logger--------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    handler = RotatingFileHandler(
        "igsp.log",
        maxBytes=1024*1024,
        backupCount=3,
        mode="a",
    )
    fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

# ---------------------- fisher_z helper -----------------------------------
from math import erf, sqrt

def fisher_z(X, Y, Z, data, boolean=False, significance_level=0.05):
    """
    Fisher‐Z conditional‐independence test.
    
    Parameters
    ----------
    X, Y : str
        Column names of the two variables.
    Z : tuple[str, ...]
        Conditioning set of column names.
    data : pd.DataFrame (or array‐like)
        Observational samples.
    boolean : bool
        If True, returns True (dependent) / False (independent) at alpha.
        If False, returns (z_stat, p_value).
    significance_level : float
        α for the 2‐sided test.
    
    Returns
    -------
    bool or (z_stat, p_value)
    """
    df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data
    x = df[X].values
    y = df[Y].values

    # 1) If Z nonempty, regress X and Y on Z to get residuals
    if Z:
        Zmat = np.column_stack([df[z].values for z in Z] + [np.ones(len(df))])
        beta_x, *_ = np.linalg.lstsq(Zmat, x, rcond=None)
        res_x = x - Zmat.dot(beta_x)
        beta_y, *_ = np.linalg.lstsq(Zmat, y, rcond=None)
        res_y = y - Zmat.dot(beta_y)
        r = np.corrcoef(res_x, res_y)[0,1]
        dof = len(x) - len(Z) - 3
    else:
        r = np.corrcoef(x, y)[0,1]
        dof = len(x) - 3

    # 2) Fisher Z‐transform
    z_stat = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(dof)

    # 3) p‐value via standard normal cdf (using erf)
    cdf = 0.5 * (1 + erf(abs(z_stat) / sqrt(2)))
    p_value = 2 * (1 - cdf)

    if boolean:
        return p_value <= significance_level
    return z_stat, p_value



# --------------------------------------------------------------------
# 1. ci_pvalue
# --------------------------------------------------------------------
def ci_pvalue(i: str,
              j: str,
              conditional: frozenset[str],
              observation: pd.DataFrame,
              ci_test: callable,
              alpha: float) -> float:
    """
    Return the p‐value for testing Xi _||_ Xj | conditional
    via the specified ci_test (only 'fisherz' supported).
    """
    if not callable(ci_test):
        raise ImportError(f"ci_test must be callable, got {type(ci_test)}")
        
    _, p_value = ci_test(i, j, tuple(conditional), observation,
                           boolean=False, significance_level = alpha)
    return p_value


# --------------------------------------------------------------------
# 2. build_i_map
# --------------------------------------------------------------------
def build_i_map(permutation: list[str],
                observation: pd.DataFrame,
                ci_test: str,
                alpha: float = 0.05) -> dict[str, set[str]]:
    """
    Construct the minimal I-MAP G_π for the given permutation π:
    for each i before j in π, add edge i→j iff pval ≤ alpha
    when testing Xi _||_ Xj | earlier - {i}.
    Returns a dict child ↦ set(parents).
    """
    parents = {v: set() for v in permutation}

    for i, j in combinations(permutation, 2):
        earlier     = permutation[:permutation.index(j)]
        conditional = frozenset(earlier) - {i}
        pval        = ci_pvalue(i, j, conditional, observation, ci_test, alpha)
        if pval <= alpha:
            parents[j].add(i)

    return parents


# --------------------------------------------------------------------
# 3. swap_adjacent
# --------------------------------------------------------------------
def swap_adjacent(permutation: list[str], k: int) -> list[str]:
    """
    Return a NEW permutation with positions k and k+1 swapped.
    """
    new = permutation.copy()
    new[k], new[k+1] = new[k+1], new[k]
    return new


# --------------------------------------------------------------------
# 4. is_i_covered
# --------------------------------------------------------------------
def is_i_covered(
    u: str,
    v: str,
    graph: dict[str, set[str]],
    intervention_targets: dict[frozenset[str], pd.DataFrame]
) -> bool:
    """
    True iff edge u→v is I-covered:
      (1) not I-contradictory;
      (2) parents(v) - {u} ⊆ parents(u).
    """
    if is_i_contradictory(u, v, intervention_targets):
        return False
    pa_v_minus_u = graph[v] - {u}
    pa_u         = graph[u]
    return pa_v_minus_u.issubset(pa_u)


# --------------------------------------------------------------------
# 5. is_i_contradictory
# --------------------------------------------------------------------
def is_i_contradictory(
    u: str,
    v: str,
    intervention_targets: dict[frozenset[str], pd.DataFrame]
) -> bool:
    """
    True ⇔ ∃ intervention target I with u∈I and v∉I.
    Only the dict’s KEYS (frozensets) are inspected.
    """
    return any((u in I) and (v not in I) for I in intervention_targets)


# --------------------------------------------------------------------
# 6. igsp_algorithm
# --------------------------------------------------------------------
def igsp_algorithm(
    observation: pd.DataFrame,
    interventions: dict[frozenset[str], pd.DataFrame],
    *,
    ci_test: Literal["fisherz"] = "fisherz",
    alpha: float = 0.05,
    verbose: bool = False
) -> set[tuple[str, str]]:
    """
    Greedy Interventional Greedy Sparsest Permutation (IGSP):
    Repeatedly reverse the best adjacent I-covered edge
    until no strictly sparser I-MAP can be found.
    Returns a set of directed‐edge tuples (u, v).
    """

    # 1. Initialize
    permutation = list(observation.columns)
    graph       = build_i_map(permutation, observation, ci_test, alpha)

    while True:
        # current edge‐count
        curr_ec = sum(len(ch) for ch in graph.values())

        # collect all adjacent I-covered reversals in a heap
        heap = []
        for k in range(len(permutation) - 1):
            u, v = permutation[k], permutation[k + 1]

            if v in graph[u] and is_i_covered(u, v, graph, interventions):
                cand_perm  = swap_adjacent(permutation, k)
                cand_graph = build_i_map(cand_perm, observation, ci_test, alpha)
                ecount     = sum(len(ch) for ch in cand_graph.values())
                contra     = is_i_contradictory(u, v, interventions)

                # key = (|E|, not-contradicting, index) for lexicographic min
                key = (ecount, not contra, k)
                heapq.heappush(heap, (key, cand_perm, cand_graph))

        # no candidates ⇒ done
        if not heap:
            break

        # pick the best reversal
        best_key, best_permutation, best_graph = heapq.heappop(heap)
        best_ecount, _, best_k = best_key

        # stop if no strict improvement
        if best_ecount >= curr_ec:
            break

        # accept and continue
        permutation, graph = best_permutation, best_graph
        if verbose:
            u_node = permutation[best_k]
            v_node = permutation[best_k + 1]
            logger.info(f"Reversed {u_node}->{v_node}, new |E|={best_ecount}")

    # emit final DAG as a set of (u, v) edges
    i_mec = {(u, v) for u, children in graph.items() for v in children}
    return i_mec

# -------------------------------- Example -----------------------------------
np.random.seed(0)
n = 100

# --- env 0: baseline observational A→B→C→… with a bit of nonlinearity in G
A0 = np.random.normal(0, 1, size=n)
B0 = 1.2 * A0 + np.random.normal(0, 0.2, size=n)
C0 = -0.8 * B0 + np.random.normal(0, 0.2, size=n)
D0 = (A0 + B0 > 0).astype(int)
E0 = C0 + 0.5 * D0 + np.random.normal(0, 0.2, size=n)
F0 = A0 - 0.5 * C0 + np.random.normal(0, 0.2, size=n)
G0 = np.sin(B0) + np.random.normal(0, 0.2, size=n)

# --- env 1: do(A)
A1 = np.random.normal(3, 1, size=n)
B1 = 1.2 * A1 + np.random.normal(0, 0.2, size=n)
C1 = -0.8 * B1 + np.random.normal(0, 0.2, size=n)
D1 = (A1 + B1 > 3).astype(int)
E1 = C1 + 0.5 * D1 + np.random.normal(0, 0.2, size=n)
F1 = A1 - 0.5 * C1 + np.random.normal(0, 0.2, size=n)
G1 = np.sin(B1) + np.random.normal(0, 0.2, size=n)

# --- env 2: do(B)
A2 = np.random.normal(0, 1, size=n)
B2 = np.random.normal(2, 1, size=n)
C2 = -0.8 * B2 + np.random.normal(0, 0.2, size=n)
D2 = (A2 + B2 > 1).astype(int)
E2 = C2 + 0.5 * D2 + np.random.normal(0, 0.2, size=n)
F2 = A2 - 0.5 * C2 + np.random.normal(0, 0.2, size=n)
G2 = np.sin(B2) + np.random.normal(0, 0.2, size=n)

# --- env 3: do({C, D})
A3 = np.random.normal(0, 1, size=n)
B3 = 1.2 * A3 + np.random.normal(0, 0.2, size=n)
E3 = 0.5 * B3 + np.random.normal(0, 0.2, size=n)
F3 = A3 + 0.3 * B3 + np.random.normal(0, 0.2, size=n)
G3 = np.cos(A3) + np.random.normal(0, 0.2, size=n)

# --- env 4: do(E)
A4 = np.random.normal(0, 1, size=n)
B4 = 1.2 * A4 + np.random.normal(0, 0.2, size=n)
C4 = -0.8 * B4 + np.random.normal(0, 0.2, size=n)
D4 = (A4 + B4 > 0).astype(int)
E4 = np.random.normal(1, 1, size=n)
F4 = A4 - 0.5 * C4 + np.random.normal(0, 0.2, size=n)
G4 = np.sin(B4) + np.random.normal(0, 0.2, size=n)

# --- env 5: do({F, G})
A5 = np.random.normal(0, 1, size=n)
B5 = 1.2 * A5 + np.random.normal(0, 0.2, size=n)
C5 = -0.8 * B5 + np.random.normal(0, 0.2, size=n)
D5 = (A5 + B5 > 0).astype(int)
E5 = C5 + 0.5 * D5 + np.random.normal(0, 0.2, size=n)
F5 = np.random.normal(-2, 1, size=n)
G5 = np.random.normal(0, 1, size=n)

# --- env 6: noise‐spike environment
A6 = np.random.normal(0, 1, size=n)
B6 = 1.2 * A6 + np.random.normal(0, 0.5, size=n)
C6 = -0.8 * B6 + np.random.normal(0, 0.5, size=n)
D6 = (A6 + B6 > 1).astype(int)
E6 = C6 * D6 + np.random.normal(0, 0.5, size=n)
F6 = np.log(np.abs(A6) + 1) + np.random.normal(0, 0.2, size=n)
G6 = np.sin(B6) + np.cos(C6) + np.random.normal(0, 0.2, size=n)

# --- concatenate all env‐blocks
df = pd.concat([
    pd.DataFrame({'env': 0, 'A':A0, 'B':B0, 'C':C0, 'D':D0, 'E':E0, 'F':F0, 'G':G0}),
    pd.DataFrame({'env': 1, 'A':A1, 'B':B1, 'C':C1, 'D':D1, 'E':E1, 'F':F1, 'G':G1}),
    pd.DataFrame({'env': 2, 'A':A2, 'B':B2, 'C':C2, 'D':D2, 'E':E2, 'F':F2, 'G':G2}),
    pd.DataFrame({'env': 3, 'A':A3, 'B':B3,             'E':E3, 'F':F3, 'G':G3}),
    pd.DataFrame({'env': 4, 'A':A4, 'B':B4, 'C':C4, 'D':D4, 'E':E4, 'F':F4, 'G':G4}),
    pd.DataFrame({'env': 5, 'A':A5, 'B':B5, 'C':C5, 'D':D5, 'E':E5, 'F':F5, 'G':G5}),
    pd.DataFrame({'env': 6, 'A':A6, 'B':B6, 'C':C6, 'D':D6, 'E':E6, 'F':F6, 'G':G6}),
], ignore_index=True)

# assume `df` is the 700-row DataFrame with an 'env' column (0…6)

interventions: dict[frozenset[str], pd.DataFrame] = {
    # env=1 was do(A): drop A
    frozenset({'A'}): df[df['env'] == 1].drop(columns=['env', 'A']),
    # env=2 was do(B): drop B
    frozenset({'B'}): df[df['env'] == 2].drop(columns=['env', 'B']),
    # env=3 was do({C, D}): drop C and D
    frozenset({'C', 'D'}): df[df['env'] == 3].drop(columns=['env', 'C', 'D']),
    # env=4 was do(E): drop E
    frozenset({'E'}): df[df['env'] == 4].drop(columns=['env', 'E']),
    # env=5 was do({F, G}): drop F and G
    frozenset({'F', 'G'}): df[df['env'] == 5].drop(columns=['env', 'F', 'G']),
    # (we typically don’t include env=6—it’s just a noisy check-env with all columns)
}

# now call:
result = igsp_algorithm(
    observation = df[df['env']==0].drop(columns='env'),
    interventions = interventions,
    ci_test = fisher_z,
    alpha   = 0.05,
    verbose = False
)
print(result)


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    













