import pandas as pd
import numpy as np
import heapq
from collections import defaultdict
from functools import lru_cache
from itertools import permutations
from typing import Callable


class PearceKellyDynamicTopoSort:
    
    """
    Dynamic topological sort (Pearce–Kelly 1993) for cycle checking of edge 
    additions and reversals to a DAG incrementally.

    Supports:
    - Constant-time ancestor test via positional lookup.
    - Edge additions and reversals with cycle checking.
    - Topological ordering automatically updated via local shifting.

    Used in LB-MDL to ensure acyclicity after each arc operation.
    """

    def __init__(self, variables: list[str]):
        self.vars  = variables                          # List of all nodes
        self.graph = defaultdict(set)                   # Adjacency list
        self.rev   = defaultdict(set)                   # Reverse adjacency (for reversals)
        self.order = variables.copy()                   # Topological order
        self.pos   = {v: i for i, v in enumerate(self.order)}  # Map: variable → topological index

    def _has_path(self, src: str, dst: str) -> bool:
        """
        Check if there exists a path from `src` to `dst`.
        DFS limited to this path only (not full reachability).
        
        Amortized time: close to O(1) per edge addition (sparse edge operations)
        although worst-case can be O(n)
        """
        stack = [src]
        seen = {src}
        while stack:
            u = stack.pop()
            if u == dst:
                return True
            for w in self.graph[u]:
                if w not in seen:
                    stack.append(w)
                    seen.add(w)
        return False

    def _shift_before(self, v: str, w: str):
        """
        Shift node `v` to come before `w` in topological order.
        Used to restore validity when an edge is added from `v` to `w`
        and `v` currently appears after `w`.
        
        Amortized time: proven to be between O(1) and O(logn) for insertions in sparse DAGs 
        """
        if self.pos[v] < self.pos[w]:
            return  # already in correct order

        self.order.pop(self.pos[v])
        self.order.insert(self.pos[w], v)

        # Rebuild position map
        for i in range(len(self.order)):
            self.pos[self.order[i]] = i

    def can_add_edge(self, parent: str, child: str) -> bool:
        """
        Returns True if edge `parent → child` can be added
        without creating a cycle.
        """
        if parent == child or child in self.graph[parent]:
            return False  # trivial loop or duplicate
        if self.pos[parent] <= self.pos[child]:
            return True   # topologically valid
        return not self._has_path(child, parent)  # would create a back edge?

    def can_reverse_edge(self, parent: str, child: str) -> bool:
        """
        Returns True if the edge `parent → child` can be reversed
        to `child → parent` without introducing a cycle.
        """
        if child not in self.graph[parent]:
            return False
        if self.pos[child] <= self.pos[parent]:
            return True  # already valid topo
        return not self._has_path(parent, child)

    def add_edge(self, parent: str, child: str) -> bool:
        """
        Attempt to add edge `parent → child`.
        Returns True if successful (DAG preserved).
        Updates the topological order if needed.
        """
        if parent == child or child in self.graph[parent]:
            return False  # loop or duplicate

        if self.pos[parent] > self.pos[child]:
            # would violate topological order → must check for cycle
            if self._has_path(child, parent):
                return False
            self._shift_before(parent, child)

        self.graph[parent].add(child)
        self.rev[child].add(parent)
        return True

    def reverse_edge(self, parent: str, child: str) -> bool:
        """
        Attempt to reverse edge `parent → child` → `child → parent`.
        If reversal causes a cycle, the original edge is restored.
        """
        if child not in self.graph[parent]:
            return False  # not an edge

        # Remove the original edge
        self.graph[parent].remove(child)
        self.rev[child].remove(parent)

        # Try to add reversed edge
        can_reverse = self.add_edge(child, parent)
        if not can_reverse:
            # Restore original edge if reversal failed
            self.graph[parent].add(child)
            self.rev[child].add(parent)

        return can_reverse
    
    
def make_local_mdl_scorer(df: pd.DataFrame) -> Callable[[str, frozenset[str]], float]:
    
    """
    Factory that returns a local MDL scoring function:
        local_mdl_score(child, parents)

    The returned function is bound to a fixed DataFrame `df`,
    and caches previously computed scores for efficiency.
    """
    
    sample_size = len(df)

    @lru_cache(maxsize=None)
    def local_mdl_score(child: str, parents: frozenset[str]) -> float:
        """
        Compute MDL(child | parents) = N * H(child | parents) + penalty
        
        where:
            - H is conditional entropy
            - penalty accounts for the number of CPT parameters
        """

        def conditional_entropy() -> float:
            """Estimate H(child | parents) using empirical probabilities."""
            if not parents:
                # No parents: Regular entropy of child
                probs = df[child].value_counts(normalize=True)
                return -(probs * np.log2(probs)).sum()

            # Compute conditional entropy: P(child | parents)
            grouped = df.groupby(list(parents))[child].value_counts(normalize=True)
            grouped = grouped.rename("p_cond").reset_index()

            # Joint Probability p(child, parents)
            joint = (
                df[list(parents) + [child]]
                .value_counts(normalize=True)
                .reset_index(name="p_joint")
            )

            # Merge conditional and joint
            grouped = grouped.merge(joint, on=list(parents) + [child], how="left")
            grouped["p_joint"] = grouped["p_joint"].fillna(0)

            # Filter out zero probabilities to avoid log(0) (convention: 0*log(0)=0)
            valid = (grouped["p_joint"] > 0) & (grouped["p_cond"] > 0)

            # H(child | parents) = -Σ p(x, u) log p(x | u)
            return -np.sum(grouped.loc[valid, "p_joint"] * np.log2(grouped.loc[valid, "p_cond"]))

        def penalty_length() -> float:
            """
            Complexity penalty:
                0.5 * log2(N) * (r - 1) * q

            where:
                - r = number of values child can take
                - q = number of distinct parent configurations
            """
            r = df[child].nunique()
            q = np.prod([df[p].nunique() for p in parents]) if parents else 1
            return 0.5 * np.log2(sample_size) * (r - 1) * q

        # MDL = N * H(child | parents) + penalty
        return sample_size * conditional_entropy() + penalty_length()

    return local_mdl_score

# ------------------- LB-MDL Bayesian Network Algorithm --------------------------
def lb_mdl_algorithm(data: pd.DataFrame, *, max_parents: int = 1) -> dict[frozenset[str], str]:
    
    """
    Learn a Bayesian Network structure using the Local Best Minimum Description Length (LB-MDL) algorithm.
    
    This algorithm implements the greedy score-based structure search of Lam & Bacchus (1994),
    which finds a directed acyclic graph (DAG) that minimises the Minimum Description Length (MDL)
    of the data. The MDL score balances model complexity and data fit by encoding both the structure
    and the parameters of the network.
    
    At each iteration, the algorithm locally adds or reverses the arc that most reduces
    the MDL score, stopping when no further local improvement is possible. Lam & Bacchus
    proved that this greedy arc-absorption process causes MDL to decrease monotonically.
    
    This implementation uses:
    
    - **MDL score caching**, exploiting the decomposable nature of the MDL objective.
      Each node’s local MDL score is stored as a function of its current parent set,
      allowing fast reuse across graph updates.
      
    - **Pearce–Kelly dynamic topological sort**, to efficiently maintain a valid topological
      ordering after each edge addition or reversal without recomputing from scratch.
    
    Parameters
    ----------
    data : pd.DataFrame
        A fully observed discrete dataset. Each column represents a categorical variable.
    * : TYPE
        (No unnamed positional arguments beyond `data` are accepted.)
    max_parents : int, optional
        Maximum number of parents allowed per node (default is 1). This restricts the parent
        search space and helps control overfitting.
    
    Returns
    -------
    dict[frozenset[str], str]
        A representation of the learned DAG structure as a dictionary mapping
        each child node to its parent set. Each key is a `frozenset` of parent variable names,
        and each value is the corresponding child node.
    
    Notes
    -----
    - The algorithm is local and greedy, so it may converge to a locally optimal structure.
      For best results, use with multiple random restarts and/or post-pruning.
    - Topological ordering is dynamically maintained during edge operations.
    - Score caching significantly improves performance for large networks.
    """

    
     # Extract node names and initialize DAG structures
    nodes = list(data.columns)
    dynamic_topo_sort = PearceKellyDynamicTopoSort(nodes)   # Maintains acyclic ordering dynamically
    parents = {node: set() for node in nodes}               # Track parent sets for each node
    local_mdl = make_local_mdl_scorer(data)                 # Factory function: returns MDL scoring function
    dag_version = 0                                         # Increments to track graph changes (for heap freshness)

    # Phase 1: Greedily add arcs that strictly reduce the total MDL score
    while True:
        valid_edges_score_heap = []
        
        # Evaluate all possible edge insertions
        for x, y in permutations(nodes, 2):
            if x in parents[y]:
                continue  # Skip if edge already exists
            if len(parents[y]) >= max_parents:
                continue  # Respect parent cap
            if not dynamic_topo_sort.can_add_edge(x, y):
                continue  # Would form a cycle

            # Compute MDL gain from adding x → y
            new_parents = frozenset(parents[y] | {x})
            old_parents = frozenset(parents[y])
            delta_mdl = local_mdl(y, new_parents) - local_mdl(y, old_parents)

            # Only keep improving candidates in heap (min-heap on delta_mdl)
            if delta_mdl < 0:
                heapq.heappush(valid_edges_score_heap, (delta_mdl, x, y, dag_version))

        # Try to add the best edge from heap
        improved = False
        while valid_edges_score_heap:
            delta_mdl, x, y, version = heapq.heappop(valid_edges_score_heap)
            if version != dag_version:
                continue  # Skip stale heap entry

            # Re-check legality due to possible concurrent updates
            if x in parents[y] or len(parents[y]) >= max_parents:
                continue
            if not dynamic_topo_sort.can_add_edge(x, y):
                continue

            # Recompute MDL gain to ensure it's still valid
            new_parents = frozenset(parents[y] | {x})
            old_parents = frozenset(parents[y])
            delta_confirm = local_mdl(y, new_parents) - local_mdl(y, old_parents)
            if delta_confirm >= 0:
                continue  # No longer an improvement

            # Apply edge: update graph, parent set, and version
            dynamic_topo_sort.add_edge(x, y)
            parents[y].add(x)
            dag_version += 1
            improved = True
            break

        if improved:
            continue  # Repeat Phase 1 on updated DAG

        # Phase 2: Try reversing edges if that reduces MDL
        reversed_any = False
        for parent in list(dynamic_topo_sort.graph.keys()):
            for child in list(dynamic_topo_sort.graph[parent]):

                # Skip if reversal would exceed parent cap
                if len(parents[parent]) + 1 > max_parents:
                    continue
                if not dynamic_topo_sort.can_reverse_edge(parent, child):
                    continue

                # Compute MDL delta from reversing parent → child to child → parent
                oldP_parent = frozenset(parents[parent])
                oldP_child  = frozenset(parents[child])
                newP_parent = frozenset(parents[parent] | {child})
                newP_child  = frozenset(parents[child]  - {parent})

                delta_rev = (
                    local_mdl(parent, newP_parent) - local_mdl(parent, oldP_parent) +
                    local_mdl(child,  newP_child)  - local_mdl(child,  oldP_child)
                )
                if delta_rev >= 0:
                    continue  # No improvement from reversal

                # Apply reversal if valid and improves the MDL score
                if dynamic_topo_sort.reverse_edge(parent, child):
                    parents[child].remove(parent)
                    parents[parent].add(child)
                    reversed_any = True

        if reversed_any:
            dag_version += 1
            continue  # After first-found reversal, return to Phase 1

        break  # No additions or reversals possible → local MDL optimum reached

    # Final DAG: a dictionary of parents -> child
    return {frozenset(parents[y]): y for y in nodes}



# Example
if __name__ == "__main__":
    
    rng = np.random.default_rng(42)
    n = 1_000            # rows
    k = 20               # columns
    
    # --- independent baselines ---------------------------------------------------
    X0 = rng.integers(0, 3, n)            # uniform 0/1/2
    X1 = rng.integers(0, 3, n)            # uniform 0/1/2
    
    # --- skewed / near-constant ---------------------------------------------------
    X2 = rng.choice([0, 1], size=n, p=[0.95, 0.05])   # 95 % zeros
    X15 = np.zeros(n, dtype=int)                      # truly constant column
    
    # --- deterministic function of parents ---------------------------------------
    X3 = (X0 + X1) % 3                                # deterministic modulo
    
    # --- noisy copy (1 % noise) ---------------------------------------------------
    noise_mask = rng.random(n) < 0.01
    X4 = X3.copy()
    X4[noise_mask] = rng.integers(0, 3, noise_mask.sum())
    
    # --- high-cardinality column w/ partial const section -------------------------
    X5 = rng.integers(0, 10, n)
    X5[900:] = 0                                      # last 100 rows constant 0
    
    # --- rare 5-th category -------------------------------------------------------
    X6 = rng.integers(0, 4, n)                        # 0–3 common
    rare_idx = rng.choice(n, 5, replace=False)
    X6[rare_idx] = 4                                  # category “4” appears 5×
    
    # --- parent-dependent categorical (mimic causal effect) ----------------------
    X7 = np.where(X2 == 1, rng.integers(0, 2, n), rng.integers(2, 4, n))
    
    # --- perfect duplicate (redundancy) ------------------------------------------
    X8 = X7.copy()
    
    # --- independent binary -------------------------------------------------------
    X9  = rng.integers(0, 2, n)
    
    # --- more independents --------------------------------------------------------
    X10 = rng.integers(0, 3, n)
    X11 = rng.integers(0, 5, n)
    
    # --- XOR-like v-structure -----------------------------------------------------
    X12 = (X2 ^ X9)  # values 0/1
    
    # --- chain dependency ---------------------------------------------------------
    X13 = (X12 + X7) % 3
    X14 = (X13 + X0) % 3
    
    # --- fill out remaining columns with mild variety ----------------------------
    X16 = rng.integers(0, 4, n)
    X17 = rng.integers(0, 6, n)
    X18 = rng.integers(0, 3, n)
    X19 = rng.integers(0, 2, n)
    
    # --- assemble DataFrame -------------------------------------------------------
    df = pd.DataFrame({
        "X0": X0,   "X1": X1,   "X2": X2,   "X3": X3,   "X4": X4,
        "X5": X5,   "X6": X6,   "X7": X7,   "X8": X8,   "X9": X9,
        "X10": X10, "X11": X11, "X12": X12, "X13": X13, "X14": X14,
        "X15": X15, "X16": X16, "X17": X17, "X18": X18, "X19": X19,
    })
    
    result = lb_mdl_algorithm(df)
    print(result)

    
    
    
    
    
    
    
    
    
    
    
    
    



















