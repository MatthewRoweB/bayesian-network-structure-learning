from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pandas as pd
import numpy as np

# ---------- Nodes ----------
@dataclass
class VaryNode:
    var: str
    mcv: Any
    children: Dict[Any, "ADNode"] = field(default_factory=dict)    # non-MCV branches only
    non_mcv_sum: int = 0                                           # Σ child.count (explicit)

@dataclass
class ADNode:
    count: int
    vary: Dict[str, VaryNode] = field(default_factory=dict)        # var -> VaryNode

# ---------- AD-Tree ----------
class ADTree:
    """
    AD-Tree with MCV compression (MCV subtree omitted).
    Public:
      - count(query: Dict[str, Any]) -> int
      - observed_parent_configs(parents: Sequence[str]) -> list[dict]
    Build knobs:
      - k: int = 3            # OS-BN sparse parent cap
      - min_count: int = 1    # Moore-style early stop threshold
      -> max_depth is set to k + 1 internally (child + up to k parents)
    """
    def __init__(
        self,
        df: pd.DataFrame,
        var_order: Optional[List[str]] = None,
        *,
        k: int = 3,
        min_count: int = 1,
    ) -> None:
        self._df = df.copy(deep=False)
        for c in self._df.columns:
            if not isinstance(self._df[c], pd.Categorical):
                cats = sorted(self._df[c].unique().tolist())
                self._df[c] = pd.Categorical(self._df[c], categories=cats, ordered=False)

        self.var_order: List[str] = var_order if var_order is not None else list(self._df.columns)
        self.max_depth: int = k + 1
        self.min_count: int = int(min_count)

        self.root = ADNode(count=len(self._df))
        self._build(self.root, self._df.index, self.var_order, depth=0)

        self.arities = {col: self._df[col].nunique() for col in self._df.columns}
        self._memo: Dict[Tuple[int, int, Tuple[Tuple[str, Any], ...]], int] = {}

    def _build(self, node: ADNode, rows: pd.Index, remaining: List[str], depth: int) -> None:
        if depth >= self.max_depth:
            return
        if len(rows) < self.min_count:
            return
        if not remaining or len(rows) == 0:
            return

        for j, var in enumerate(remaining):
            s = self._df.loc[rows, var]
            counts = s.value_counts(sort=False)
            if counts.empty:
                continue

            mcv = counts.idxmax()
            vnode = VaryNode(var=var, mcv=mcv)
            node.vary[var] = vnode

            for val, cnt in counts.items():
                if val == mcv or cnt == 0:
                    continue
                if cnt < self.min_count:
                    continue
                child_rows = rows[s == val]
                child = ADNode(count=int(cnt))
                vnode.children[val] = child
                self._build(child, child_rows, remaining[j + 1:], depth=depth + 1)

            vnode.non_mcv_sum = sum(ch.count for ch in vnode.children.values())

    def count(self, query: Dict[str, Any]) -> int:
        key = (id(self.root), 0, tuple(sorted(query.items())))
        hit = self._memo.get(key)
        if hit is not None:
            return hit
        out = self._count(self.root, query, 0)
        self._memo[key] = out
        return out

    def _count(self, node: ADNode, query: Dict[str, Any], start: int) -> int:
        if not query:
            return node.count

        for i in range(start, len(self.var_order)):
            var = self.var_order[i]
            if var not in query:
                continue

            vnode = node.vary.get(var)
            if vnode is None:
                return 0

            mkey = (id(node), i, tuple(sorted(query.items())))
            hit = self._memo.get(mkey)
            if hit is not None:
                return hit

            val = query[var]
            rest = dict(query); rest.pop(var)

            if val == vnode.mcv:
                total = self._count(node, rest, i)
                for child in vnode.children.values():
                    total -= self._count(child, rest, i + 1)
                self._memo[mkey] = total
                return total
            else:
                child = vnode.children.get(val)
                if child is None:
                    self._memo[mkey] = 0
                    return 0
                ans = self._count(child, rest, i + 1)
                self._memo[mkey] = ans
                return ans

        return node.count

    def observed_parent_configs(self, parents: Sequence[str]) -> list[dict]:
        if not parents:
            return [{}]
        pset = set(parents)
        order = [v for v in self.var_order if v in pset]
        if not order:
            return [{}]

        results: list[dict] = []
        partial: dict = {}

        def dfs(node: ADNode, k: int) -> None:
            if k == len(order):
                if self.count(partial) > 0:
                    results.append(partial.copy())
                return
            var = order[k]
            vnode = node.vary.get(var)
            if vnode is None:
                return
            for val, child_node in vnode.children.items():
                partial[var] = val
                dfs(child_node, k + 1)
                partial.pop(var, None)
            if node.count - vnode.non_mcv_sum > 0:
                partial[var] = vnode.mcv
                dfs(node, k + 1)
                partial.pop(var, None)

        dfs(self.root, 0)
        return results


# --------------------- Order-Search BN helpers ----------------------------
import heapq
from heapdict import heapdict
from sklearn.metrics import mutual_info_score
from itertools import combinations
from scipy.special import gammaln


def top_k_mutual_info_per_node(df: pd.DataFrame, k: int) -> dict[str, set]:
    """
    Per-node top-k MI candidates (Teyssier & Koller, 2005).
    Uses a shared MI cache, generator, and heapq.nlargest.
    Returns: allowed_parents: var -> set of top-k most related variables (by MI).
    """
    cols = df.columns.to_list()
    arr = df.to_numpy()

    # shared MI cache across all variables
    mi_cache: dict[frozenset[str], float] = {}
    allowed_parents: dict[str, set] = {}

    for i, xi in enumerate(cols):
        # generator of (xj, MI(xi, xj)) using a symmetric cache
        scores_gen = (
            (
                xj,
                mi_cache.setdefault(
                    frozenset((xi, xj)),
                    mutual_info_score(arr[:, i], arr[:, j])
                )
            )
            for j, xj in enumerate(cols) if j != i
        )

        # always compute MI and select top-k, even if k >= n-1
        topk = heapq.nlargest(k, scores_gen, key=lambda t: t[1])

        # store only the variables (parents) in a set
        allowed_parents[xi] = {xj for (xj, _) in topk}

    return allowed_parents

def local_bdeu_score(child: str, parents: Sequence[str], ess: float, adtree: ADTree) -> float:
    r_x = adtree.arities[child]
    q = int(np.prod([adtree.arities[p] for p in parents])) if parents else 1
    alpha_j  = ess / q
    alpha_jk = ess / (q * r_x)

    log_score = 0.0
    for j_dict in adtree.observed_parent_configs(parents):
        N_j = adtree.count(j_dict)
        term = gammaln(alpha_j) - gammaln(alpha_j + N_j)
        for k in adtree._df[child].cat.categories:
            N_jk = adtree.count({**j_dict, child: k})
            term += gammaln(alpha_jk + N_jk) - gammaln(alpha_jk)
        log_score += term
    return float(log_score)

def choose_best_parents(child: str,
                        cands: Sequence[str],
                        k: int,
                        *,
                        local_score=local_bdeu_score,
                        ess: float = 1.0,
                        adtree: ADTree) -> tuple[frozenset[str], float]:
    """
    Enumerate P ⊆ cands with |P| ≤ k using local dominance pruning.
    Return (best_parent_set, best_local_score).
    """
    cands = tuple(cands)
    max_r = min(k, len(cands))

    best_P = frozenset()
    best_S = local_score(child, best_P, ess=ess, adtree=adtree)

    for r in range(1, max_r + 1):
        for P_tuple in combinations(cands, r):
            P = frozenset(P_tuple)
            sP = local_score(child, P, ess=ess, adtree=adtree)

            # prune by immediate-subset dominance
            dominated = False
            for p in P:
                Q = frozenset(P - {p})
                if local_score(child, Q, ess=ess, adtree=adtree) >= sP:
                    dominated = True
                    break
            if dominated:
                continue

            if sP > best_S:
                best_P, best_S = P, sP

    return best_P, best_S


# --------------------- Main OS Algorithm (with heapdict) -------------------

def order_search_algorithm(
    data: pd.DataFrame,
    *,
    max_parents: int = 3,
    ess: float = 1.0,
    top_scores: int = 5,
    random_restarts: int = 5,
    random_state: Optional[int] = None,
) -> tuple[Dict[str, frozenset[str]], float]:
    """
    Ordering-based search (OS) using BDeu and greedy adjacent swaps.
    Returns (best_dag: node->parents, best_score).
    """
    rng = np.random.default_rng(seed=random_state)
    nodes: List[str] = data.columns.to_list()
    n = len(nodes)

    adtree = ADTree(data) 
    allowed_par = top_k_mutual_info_per_node(data, k=top_scores)

    best_global_dag: Dict[str, frozenset[str]] = {}
    best_global_score = -float("inf")

    for _ in range(random_restarts):
        ordered_nodes = nodes.copy()
        rng.shuffle(ordered_nodes)

        # --- initial local scores per position ---
        parents_by_pos = [frozenset()]*n
        score_by_pos   = [0.0]*n
        
        for pos, node in enumerate(ordered_nodes):
            preds = set(ordered_nodes[:pos])
            cands = preds & allowed_par.get(node, set())
            bp, bs = choose_best_parents(node, cands, k=max_parents,
                                         local_score=local_bdeu_score,
                                         ess=ess, adtree=adtree)
            parents_by_pos[pos] = bp
            score_by_pos[pos] = bs

        # --- Δ helper for an adjacent swap (j, j+1) ---
        def delta_for_swap(j: int) -> float:
            a, b = ordered_nodes[j], ordered_nodes[j + 1]
            old_a, old_b = score_by_pos[j], score_by_pos[j + 1]

            # After swap: b moves earlier, a moves later.
            preds_b = set(ordered_nodes[:j])          # b's predecessors after swap
            preds_a = set(ordered_nodes[:j + 1])      # a's predecessors after swap (includes b)

            cands_b = preds_b & allowed_par.get(b, set())
            cands_a = preds_a & allowed_par.get(a, set())

            _, new_b = choose_best_parents(b, cands_b, k=max_parents,
                                           local_score=local_bdeu_score,
                                           ess=ess, adtree=adtree)
            _, new_a = choose_best_parents(a, cands_a, k=max_parents,
                                           local_score=local_bdeu_score,
                                           ess=ess, adtree=adtree)
            return (new_a + new_b) - (old_a + old_b)

        # --- build initial heapdict with improving swaps only ---
        hd = heapdict()
        for j in range(n - 1):
            d = delta_for_swap(j)
            if d > 0:
                hd[j] = (-d, j)  # store as min-heap priority

        # --- greedy adjacent-swap loop ---
        while hd:
            j, (negd, _) = hd.popitem()
            d = -negd  # strictly > 0

            # apply the swap in the ordering
            ordered_nodes[j], ordered_nodes[j + 1] = ordered_nodes[j + 1], ordered_nodes[j]

            # recompute locals for the two swapped positions
            for idx in (j, j + 1):
                node = ordered_nodes[idx]
                preds = set(ordered_nodes[:idx])
                cands = preds & allowed_par.get(node, set())
                bp, bs = choose_best_parents(node, cands, k=max_parents,
                                             local_score=local_bdeu_score,
                                             ess=ess, adtree=adtree)
                parents_by_pos[idx] = bp
                score_by_pos[idx] = bs

            # refresh only neighbors j-1, j, j+1
            for k in (j - 1, j, j + 1):
                if 0 <= k < n - 1:
                    dk = delta_for_swap(k)
                    if dk > 0:
                        hd[k] = (-dk, k)   # insert/update if improving
                    else:
                        hd.pop(k, None)    # remove if present and non-improving

        # local optimum reached for this restart
        total = sum(score_by_pos)
        if total > best_global_score:
            best_global_score = total
            best_global_dag = {ordered_nodes[pos]: parents_by_pos[pos] for pos in range(n)}

    return best_global_dag, best_global_score



#  ------------- Testing -------------------------------
import numpy as np
import pandas as pd

# ---- ALARM-like variables & arities (0..r-1) ----
ARITY = {
    "CVP":3, "PCWP":3, "HIST":2, "TPR":3, "BP":3, "CO":3,
    "HRBP":3, "HREK":3, "HRSA":3, "PAP":3, "SAO2":3, "FIO2":2,
    "PRSS":4, "ECO2":4, "MINV":4, "MVS":3, "HYP":2, "LVF":2,
    "APL":2, "ANES":2, "PMB":2, "INT":3, "KINK":2, "DISC":2,
    "LVV":3, "STKV":3, "CCHL":2, "ERLO":2, "HR":3, "ERCA":2,
    "SHNT":2, "PVS":3, "ACO2":3, "VALV":4, "VLNG":4, "VTUB":4, "VMCH":4
}

# A handy topological order (parents first). This is just one consistent choice.
ORDER = [
    # roots
    "HIST","HYP","LVF","ANES","FIO2","APL","PMB","CCHL","ERLO","ERCA",
    "INT","KINK","DISC","VALV","VLNG","VTUB","VMCH",
    # cardio/mech derived
    "TPR","CO","CVP","PCWP","HR","HRBP","HREK","HRSA","BP","PVS","LVV","STKV",
    # ventilation chain
    "MINV","MVS","PRSS","ECO2","ACO2","PAP","SAO2"
]

def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(axis=1, keepdims=True)
    return z

def _sample_categorical(probs, rng):
    # probs: (N, r)
    cdf = np.cumsum(probs, axis=1)
    u = rng.random((probs.shape[0], 1))
    return (u > cdf).sum(axis=1).astype(np.int64)

def _normalize_state(x, r):
    # map 0..r-1 -> [0,1]
    return x / max(1, (r - 1))

def make_alarm_like_df(n_rows=10_000, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    N = n_rows
    X = {}

    # ---- Roots: base prevalences ----
    # rare-ish conditions
    for v, p in {
        "HIST":0.2, "HYP":0.25, "LVF":0.15, "ANES":0.4, "FIO2":0.6,
        "APL":0.1, "PMB":0.1, "CCHL":0.05, "ERLO":0.05, "ERCA":0.05
    }.items():
        r = ARITY[v]
        # binary roots ~ Bernoulli; multi-state roots ~ slightly biased categorical
        if r == 2:
            X[v] = (rng.random(N) < p).astype(np.int64)
        else:
            probs = rng.dirichlet(np.ones(r)) * 0.6 + np.eye(r)[0] * 0.4
            probs = probs / probs.sum()
            X[v] = rng.choice(r, size=N, p=probs)

    # airway/vent hardware roots (mostly normal)
    for v, r in [("INT",3),("KINK",2),("DISC",2),("VALV",4),("VLNG",4),("VTUB",4),("VMCH",4)]:
        base = np.array([0.7, 0.2, 0.1] + [])[0:r]  # start with skew to state 0
        if r == 2: probs = np.array([0.9, 0.1])
        elif r == 3: probs = np.array([0.8, 0.15, 0.05])
        elif r == 4: probs = np.array([0.8, 0.12, 0.06, 0.02])
        probs = probs / probs.sum()
        X[v] = rng.choice(r, size=N, p=probs)

    # ---- Cardiovascular block ----
    # TPR (3): rises with HYP, falls with ANES
    z = np.zeros((N, 3))
    z[:,1] = 0.0  # normal baseline
    z[:,0] = -0.5 + 0.7*_normalize_state(X["ANES"], ARITY["ANES"])      # low TPR if anesthesia
    z[:,2] = -0.6 + 1.2*_normalize_state(X["HYP"], ARITY["HYP"])        # high TPR if hypovolemia
    probs = _softmax(z + rng.normal(0, 0.15, z.shape))
    X["TPR"] = _sample_categorical(probs, rng)

    # CO (3): lower with HYP and LVF; slightly higher with high FIO2
    z = np.zeros((N, 3))
    z[:,1] = 0.0
    hyp = _normalize_state(X["HYP"], 2)
    lvf = _normalize_state(X["LVF"], 2)
    fio = _normalize_state(X["FIO2"], 2)
    z[:,0] = -0.3 + 1.1*hyp + 1.0*lvf          # low CO if hypovolemia or LV failure
    z[:,2] = -0.8 + 0.4*fio                    # high CO a bit more likely with good oxygen
    probs = _softmax(z + rng.normal(0, 0.15, z.shape))
    X["CO"] = _sample_categorical(probs, rng)

    # CVP (3): low with HYP, high with LVF
    z = np.zeros((N, 3)); z[:,1]=0.0
    z[:,0] = -0.3 + 1.1*hyp
    z[:,2] = -0.6 + 1.2*lvf
    X["CVP"] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # PCWP (3): rises with LVF
    z = np.zeros((N, 3)); z[:,1]=0.0
    z[:,2] = -0.8 + 1.6*lvf
    X["PCWP"] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # HR (3): higher with HYP, ANES can lower slightly
    z = np.zeros((N, 3)); z[:,1]=0.0
    z[:,0] = -0.4 + 0.6*_normalize_state(X["ANES"],2)
    z[:,2] = -0.5 + 1.2*hyp
    X["HR"] = _sample_categorical(_softmax(z + rng.normal(0, 0.15, z.shape)), rng)

    # HR-* lead derivations (mildly dependent on HR)
    for v in ["HRBP","HREK","HRSA"]:
        r = ARITY[v]
        z = np.zeros((N, r))
        for s in range(r):
            z[:,s] = -0.5*abs(s - _normalize_state(X["HR"],3)*2.0)
        X[v] = _sample_categorical(_softmax(z + rng.normal(0, 0.15, z.shape)), rng)

    # BP (3): increases with TPR, decreases with CO (low CO → low BP)
    z = np.zeros((N, 3)); z[:,1]=0.0
    tpr = _normalize_state(X["TPR"],3)
    co  = _normalize_state(X["CO"],3)
    z[:,0] = -0.4 + 1.0*(1.0 - co)   # low BP if CO is low
    z[:,2] = -0.6 + 1.0*tpr          # high BP if TPR is high
    X["BP"] = _sample_categorical(_softmax(z + rng.normal(0, 0.15, z.shape)), rng)

    # Peripheral/systemic volumes (PVS, LVV, STKV) — informative of HYP/LVF
    for v, drive in [("PVS", ("HYP", 1.2)), ("LVV", ("LVF", 1.2)), ("STKV", ("HYP", 0.9))]:
        r = ARITY[v]
        par = _normalize_state(X[drive[0]], 2)
        z = np.zeros((N, r)); z[:,1 if r==3 else 0] = 0.0
        if r == 3:
            z[:,0] = -0.3 + 0.9*par
            z[:,2] = -0.7 + 0.8*(1.0 - par)
        else:
            for s in range(r):
                z[:,s] = -0.5*abs(s - par*(r-1))
        X[v] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # ---- Ventilation block ----
    # MINV (4): affected by INT, VMCH, VTUB, VALV, KINK, DISC
    def combine_faults(*vals):
        # map multiple hardware states to a [0,1] "fault severity"
        sev = 0.0
        for v, r in vals:
            sev += _normalize_state(X[v], r)
        return np.clip(sev / len(vals), 0, 1)

    vent_fault = combine_faults(("INT",3),("VMCH",4),("VTUB",4),("VALV",4),("KINK",2),("DISC",2))
    z = np.zeros((N, 4));  # 0=best .. 3=worst
    for s in range(4):
        z[:,s] = -0.8*abs(s - (1.2 + 2.0*vent_fault))
    X["MINV"] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # MVS (3): follows MINV
    z = np.zeros((N, 3))
    minv_norm = _normalize_state(X["MINV"], 4)
    for s in range(3):
        z[:,s] = -0.8*abs(s - (minv_norm * 2.0))
    X["MVS"] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # PRSS (4): pressure issues with KINK/DISC/VALV faults and high MVS
    mech_fault = combine_faults(("KINK",2),("DISC",2),("VALV",4))
    z = np.zeros((N, 4))
    mvsn = _normalize_state(X["MVS"], 3)
    for s in range(4):
        z[:,s] = -0.7*abs(s - (0.8 + 2.2*mech_fault + 0.8*mvsn))
    X["PRSS"] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # ECO2 (4): rises with poor ventilation (MINV fault) and valve/kink issues
    z = np.zeros((N, 4))
    for s in range(4):
        z[:,s] = -0.85*abs(s - (0.6 + 2.4*vent_fault))
    X["ECO2"] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # ACO2 (3): derived from ECO2
    eco2n = _normalize_state(X["ECO2"], 4)
    z = np.zeros((N, 3))
    for s in range(3):
        z[:,s] = -0.9*abs(s - (eco2n*2.0))
    X["ACO2"] = _sample_categorical(_softmax(z + rng.normal(0, 0.1, z.shape)), rng)

    # PAP (3): correlates with LVF and ventilation pressures (PRSS)
    prssn = _normalize_state(X["PRSS"], 4)
    z = np.zeros((N, 3)); z[:,1]=0.0
    z[:,2] = -0.8 + 1.0*lvf + 0.7*prssn
    X["PAP"] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # SAO2 (3): better with FIO2, worse with ventilation faults (ECO2 high) and LVF
    z = np.zeros((N, 3)); z[:,1]=0.0
    z[:,0] = -0.3 + 1.3*eco2n + 0.6*lvf
    z[:,2] = -0.6 + 1.2*fio
    X["SAO2"] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # Ensure all variables exist; if any left, make weakly informative noisy copies of related parents
    for v in ARITY:
        if v not in X:
            r = ARITY[v]
            parent = "HR" if r==3 else "CO"
            pn = _normalize_state(X[parent], ARITY[parent])
            z = np.zeros((N, r))
            for s in range(r):
                z[:,s] = -0.6*abs(s - pn*(r-1))
            X[v] = _sample_categorical(_softmax(z + rng.normal(0, 0.12, z.shape)), rng)

    # Assemble DataFrame (int codes 0..r-1)
    return pd.DataFrame({k: X[k] for k in ORDER + [v for v in ARITY if v not in ORDER]})

# ----- build data -----
df_alarm_like = make_alarm_like_df(n_rows=10_000, seed=123)
print(df_alarm_like.shape)  # (10000, 37)


import time

# params per Teyssier & Koller (ALARM benchmark)
params = dict(
    max_parents=3,
    ess=1.0,
    top_scores=5,       # per-node top-k MI
    random_restarts=10,
    random_state=42
)

t0 = time.perf_counter()
best_dag, best_score = order_search_algorithm(df_alarm_like, **params)
elapsed = time.perf_counter() - t0

print(f"Runtime: {elapsed:.2f} s")
print(f"BDeu score: {best_score:.2f}")
# (Optional) peek a few learned parents
for i, (child, parents) in enumerate(best_dag.items()):
    if i == 8: break
    print(f"{child} ← {sorted(parents)}")





















