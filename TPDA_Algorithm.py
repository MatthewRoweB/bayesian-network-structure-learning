import pandas as pd
from itertools import chain, combinations
from sklearn.metrics import mutual_info_score
from collections import defaultdict, deque
from typing import DefaultDict, Iterable, Generator
import logging
import random

logger = logging.getLogger(__name__)
logger.propagate = False          

logging.basicConfig(filename='drafting_run.log', level=logging.INFO, filemode='w')


# ---------------------------------------------------------------------------

class DisjointSet:
    def __init__(self, vertices: list[str]):
        self.parent = {v: v for v in vertices} 
        self.rank = {v: 0 for v in vertices}
        
    def find(self, vertex: str):
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
            
        return self.parent[vertex]
    
    def union(self, root1: str, root2: str):   
        if self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
            
        elif self.rank[root2] < self.rank[root1]:
            self.parent[root1] = root2
            
        else: 
            self.parent[root2] = root1
            self.rank[root1] += 1
            
def kruskal(graph: DefaultDict[str, list[str]]):    
    
    edges = []
    for vertex, neighbours in graph.items():
        for neighbour, weight in neighbours:
            if (neighbour, vertex, weight) not in edges:
                edges.append((vertex, neighbour, weight))
                
    edges.sort(key = lambda edge: (edge[2], edge[0], edge[1]))
    
    vertices = list(graph.keys())
    disjoint_set = DisjointSet(vertices)
    
    mst = []
    for u, v, weight in edges:
        # Union-Find Algorithm
        root_u = disjoint_set.find(u)
        root_v = disjoint_set.find(v)
        
        if root_u != root_v:
            mst.append((u, v, weight))
            disjoint_set.union(root_u, root_v)
            
    return mst


# ────────────────────────────────────────────────────────────────────────────
def drafting(
    data: pd.DataFrame,
    threshold: float = 1e-12,
    *,
    verbose: bool = False,
    ) -> tuple[list[tuple[str, str, float]], list[tuple[str, str, float]]]:
    """
    Phase-1 Drafting.

    Parameters
    ----------
    data     : DataFrame of discrete vars
    threshold: ignore MI ≤ threshold
    verbose  : Stream INFO to console when True
    logfile  : path to write a full INFO log (optional)

    Returns
    -------
    tree            , remaining_edges
    """

    # ---- build weighted complete graph ------------------------------------
    vars_: list[str] = list(data.columns)
    graph: dict[str, list[tuple[str, float]]] = defaultdict(list,
                                                           {v: [] for v in vars_})
    all_edges: list[tuple[str, str, float]] = []

    logger.info("Drafting: %d variables", len(vars_))

    for X, Y in combinations(vars_, 2):
        mi = mutual_info_score(data[X], data[Y])
        if mi > threshold:
            graph[X].append((Y, mi))
            graph[Y].append((X, mi))
            all_edges.append((X, Y, mi))
            logger.info("Edge kept (%s,%s)  MI=%.6g", X, Y, mi)

    logger.info("Kept %d edges with MI > %.3g", len(all_edges), threshold)

    # ---- Kruskal maximal-weight spanning tree -----------------------------
    tree: list[tuple[str, str, float]] = kruskal(graph)

    tree_set: set[tuple[str, str, float]] = set(tree)
    remaining = [e for e in all_edges if e not in tree_set]

    logger.info("MWST edges: %d   remaining edges: %d",
                len(tree), len(remaining))

    return tree, remaining

def conditional_mi(
    df: pd.DataFrame, x: str, y: str, Z: list[str]) -> float:
    """
    I(X;Y | Z) using sklearn.metrics.mutual_info_score.
    Works for Z = [] (plain MI).  Variables must be discrete.
    """
    if not Z:
        return mutual_info_score(df[x], df[y])

    n = len(df)
    cmi = 0.0
    for _, sub in df.groupby(Z, sort=False):
        weight = len(sub) / n
        cmi   += weight * mutual_info_score(sub[x], sub[y])
    return cmi

def subsets_nonempty(U: Iterable) -> Generator[tuple, None, None]:
    """
    Yield all non-empty subsets of U in size order.
    """
    U = tuple(U)           # freeze order once
    return chain.from_iterable(combinations(U, r) 
                               for r in range(1, len(U) + 1))

def has_independent_subset(
    df: pd.DataFrame,
    x: str,
    y: str,
    U: set[str],
    threshold: float) -> bool:
    """
    Return True iff ∃ non-empty Z ⊆ U with      I(X;Y|Z) ≤ threshold
    """
    return any(
        conditional_mi(df, x, y, list(Z)) <= threshold
        for Z in subsets_nonempty(U)
    )

def tarjan_bcc(adj: dict[str, set[str]]
               ) -> tuple[list[set[str]], set[str]]:
    disc, low, time = {}, {}, [0]
    edge_st, blocks, cuts = [], [], set()

    def dfs(v: str, parent: str | None = None):
        disc[v] = low[v] = time[0]; time[0] += 1
        child_cnt = 0

        for nb in adj[v]:
            if nb == parent:
                continue
            if nb not in disc:                         # tree edge
                edge_st.append((v, nb))
                child_cnt += 1
                dfs(nb, v)
                low[v] = min(low[v], low[nb])

                root_cut   = parent is None and child_cnt > 1
                nonroot_cut = parent and low[nb] >= disc[v]
                if root_cut or nonroot_cut:
                    cuts.add(v)
                    comp = set()
                    while edge_st and edge_st[-1] not in {(v, nb), (nb, v)}:
                        comp.update(edge_st.pop())
                    comp.update(edge_st.pop())
                    blocks.append(comp)
            elif disc[nb] < disc[v]:                   # back edge
                edge_st.append((v, nb))
                low[v] = min(low[v], disc[nb])

    for v in adj:
        if v not in disc:
            dfs(v)
            if edge_st:                               # leftover edges
                comp = set()
                while edge_st:
                    comp.update(edge_st.pop())
                blocks.append(comp)

    return blocks, cuts

def between_vertices(
    adj: dict[str, set[str]], src: str, dst: str
) -> set[str]:
    """
    Return all internal vertices on *some* simple src→dst path.
    If src and dst are disconnected (no simple path), return ∅.
    Handles isolated vertices by creating size-1 “trivial” blocks.
    """
    blocks, cuts = tarjan_bcc(adj)

    # map vertices → block indices
    bct: dict[str | int, set[str | int]] = defaultdict(set)
    v2blocks: dict[str, list[int]] = defaultdict(list)
    for idx, blk in enumerate(blocks):
        for v in blk:
            v2blocks[v].append(idx)

    # ensure isolated vertices still map to a block
    for v in adj:
        if v not in v2blocks:
            idx = len(blocks)
            blocks.append({v})
            v2blocks[v].append(idx)

    # build block-cut tree edges
    for cut in cuts:
        for b in v2blocks[cut]:
            bct[cut].add(b)
            bct[b].add(cut)

    # endpoints missing → disconnected
    if src not in v2blocks or dst not in v2blocks:
        return set()

    src_block = v2blocks[src][0]
    dst_block = v2blocks[dst][0]
    if src_block == dst_block:
        return set()                  # same block ⇒ no internal vertices

    # BFS along the tree
    parent = {src_block: None}
    q = deque([src_block])
    while q:
        node = q.popleft()
        if node == dst_block:
            break
        for nb in bct[node]:
            if nb not in parent:
                parent[nb] = node
                q.append(nb)
    else:                              # no route in tree
        return set()

    # reconstruct path
    path = []
    node = dst_block
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()

    U: set[str] = set()
    for node in path:
        if isinstance(node, int):      # block id
            U.update(blocks[node])
        else:                          # articulation vertex
            U.add(node)

    U.discard(src)
    U.discard(dst)
    return U

def thickening(
    data: pd.DataFrame,
    tree: list[tuple[str, str, float]],
    remaining_edges: list[tuple[str, str, float]],
    *,
    threshold: float = 1e-12,
    verbose: bool = False,
) -> tuple[set[frozenset[str]], dict[tuple[str, str], set[str]]]:
    """
    Returns
    -------
    final_edges : set[frozenset]   # undirected edges after thickening
    sepsets     : dict[(u,v)] -> set  # separating sets discovered here
    """
    sepsets: dict[tuple[str, str], set[str]] = {}   # NEW

    # mutable adjacency initialised with the Phase-1 tree
    adj: dict[str, set[str]] = defaultdict(set)
    for u, v, _ in tree:
        adj[u].add(v); adj[v].add(u)

    # iterate all remaining MI-ranked pairs once
    for X, Y, _ in remaining_edges:
        if Y in adj[X]:
            continue                      # already adjacent

        U = between_vertices(adj, X, Y)
        if not U:
            continue

        sep_found = None
        for Z in subsets_nonempty(U):
            if conditional_mi(data, X, Y, list(Z)) <= threshold:
                sep_found = Z
                break

        if sep_found is not None:        # independence ⇒ NO edge
            key = (X, Y) if X < Y else (Y, X)
            sepsets[key] = set(sep_found)
            if verbose:
                logger.info("skip (%s,%s)  — SepSet %s", X, Y, sep_found)
            continue

        # dependence ⇒ add undirected edge
        adj[X].add(Y); adj[Y].add(X)
        if verbose:
            logger.info("add  (%s,%s)", X, Y)

    # convert adjacency to frozensets
    final_edges = {
        frozenset((u, v))
        for u in adj
        for v in adj[u]
        if u < v
    }
    return final_edges, sepsets

def thinning(
    data: pd.DataFrame,
    final_edges: set[frozenset[str]],
    *,
    threshold: float = 1e-12,
    sepsets: dict[tuple[str, str], set[str]] | None = None,
    verbose: bool = False,
) -> tuple[set[frozenset[str]], dict[tuple[str, str], set[str]]]:

    if sepsets is None:
        sepsets = {}

    # build adjacency
    adj: dict[str, set[str]] = defaultdict(set)
    for e in final_edges:
        u, v = tuple(e)
        adj[u].add(v); adj[v].add(u)

    pruned = set(final_edges)            # working copy

    for edge in list(pruned):            # static list to iterate once
        x, y = tuple(edge)

        # temporarily remove edge
        adj[x].remove(y); adj[y].remove(x)
        U = between_vertices(adj, x, y)
        adj[x].add(y); adj[y].add(x)     # restore for next pair

        if not U:
            continue

        sep_found = None
        for Z in subsets_nonempty(U):
            if conditional_mi(data, x, y, list(Z)) <= threshold:
                sep_found = Z
                break

        if sep_found is not None:        # remove & record SepSet
            pruned.remove(edge)
            adj[x].remove(y); adj[y].remove(x)
            key = (x, y) if x < y else (y, x)
            sepsets[key] = set(sep_found)
            
            if verbose:
                logger.info("Removed (%s,%s) — SepSet %s", x, y, sep_found)
        else:
            if verbose:
                logger.info("Kept    (%s,%s)", x, y)

    return pruned, sepsets



# ───────────────────────────────────────────────────────────────────────────
def orient_edges(
    skeleton: set[frozenset[str]],
    sepsets : dict[tuple[str, str], set[str]],
    *,
    verbose: bool = False
    ) -> set[tuple[str, str] | frozenset[str]]:
    """
    Phase-4 orientation  (PC / Meek style).

    Parameters
    ----------
    skeleton : set[frozenset]        # undirected edges after thinning
    sepsets  : dict[(u,v)] -> set    # separating sets recorded earlier
    verbose  : bool                  # INFO to console if True

    Returns
    -------
    mixed : set[tuple | frozenset]
        Directed edges   → tuple(tail, head)
        Undirected edges → frozenset({u,v})
    """

    # ---------- build adjacency -----------------------------------------
    adj: dict[str, set[str]] = defaultdict(set)
    for e in skeleton:
        u, v = tuple(e)
        adj[u].add(v)
        adj[v].add(u)

    directed: set[tuple[str, str]]   = set()           # arrows
    undirected: set[frozenset[str]]  = set(skeleton)   # start with all

    # ---------- 1. identify colliders -----------------------------------
    for A, B, C in combinations(adj, 3):
        triples = [(A, B, C), (B, C, A), (C, A, B)]
        for X, Y, Z in triples:
            if (Y in adj[X]) and (Y in adj[Z]) and (X not in adj[Z]):
                S = sepsets.get(tuple(sorted((X, Z))), set())
                if Y not in S:
                    # orient X -> Y <- Z
                    if frozenset((X, Y)) in undirected:
                        undirected.remove(frozenset((X, Y)))
                        directed.add((X, Y))
                    if frozenset((Z, Y)) in undirected:
                        undirected.remove(frozenset((Z, Y)))
                        directed.add((Z, Y))
                        if verbose:
                            logger.info("Collider %s -> %s <- %s", X, Y, Z)

    # ---------- 2. propagate directions (Meek rules) --------------------
    changed = True
    while changed:
        changed = False

        # R1: X->Y, Y-Z undirected, X not adj Z  ⇒ orient Y->Z
        for YZ in list(undirected):
            Y, Z = tuple(YZ)
            for X in adj[Y]:
                if (X, Y) in directed and Z not in adj[X]:
                    undirected.remove(YZ)
                    directed.add((Y, Z))
                    changed = True
                    if verbose:
                        logger.info("R1: %s -> %s", Y, Z)
                    break
            if changed:
                break

        # R2: X-Y undirected, exists X->Z->Y  ⇒ X->Y
        if not changed:
            for XY in list(undirected):
                X, Y = tuple(XY)
                for Z in adj[X] & adj[Y]:
                    if (X, Z) in directed and (Z, Y) in directed:
                        undirected.remove(XY)
                        directed.add((X, Y))
                        changed = True
                        if verbose:
                            logger.info("R2: %s -> %s", X, Y)
                        break
                if changed:
                    break

        # R3: X-Y undirected, two non-adjacent Z,W with X->Z, X->W, Z,W ->Y  ⇒ X->Y
        if not changed:
            for XY in list(undirected):
                X, Y = tuple(XY)
                ZW = [Z for Z in adj[X] & adj[Y] if (X, Z) in directed]
                for i, Z in enumerate(ZW):
                    for W in ZW[i + 1:]:
                        if W not in adj[Z] and (Z, Y) in directed and (W, Y) in directed:
                            undirected.remove(XY)
                            directed.add((X, Y))
                            changed = True
                            if verbose:
                                logger.info("R3: %s -> %s", X, Y)
                            break
                    if changed:
                        break
                if changed:
                    break

        # R4: orient to avoid new collider in triangles
        if not changed:
            for XY in list(undirected):
                X, Y = tuple(XY)
                for Z in adj[X] & adj[Y]:
                    if (X, Z) in directed and frozenset((Z, Y)) in undirected:
                        undirected.remove(frozenset((Z, Y)))
                        directed.add((Z, Y))
                        changed = True
                        if verbose:
                            logger.info("R4: %s -> %s", Z, Y)
                        break
                if changed:
                    break

    # ---------- 3 merge into one mixed set -----------------------------
    mixed: set[tuple[str, str] | frozenset[str]] = set()
    mixed.update(directed)    # tuples
    mixed.update(undirected)  # frozensets

    return mixed


def TPDA_Algorithm(data: pd.DataFrame, *, 
                   threshold: float = 1e-12, 
                   verbose: bool = False) -> set[frozenset, tuple]:
    # Phase-1
    tree, rem = drafting(data, verbose=True)

    # Phase-2
    edges2, seps = thickening(
        data, tree, rem,
        verbose=True
    )

    # Phase-3
    edges3, seps = thinning(
        data, edges2,
        sepsets=seps,
        verbose=True
    )

    # Phase-4 
    cpdag = orient_edges(
        skeleton=edges3,
        sepsets=seps,
        verbose=True
    )
    return cpdag

if __name__ == '__main__':
    df = pd.DataFrame({"X": [random.randint(0,3) for _ in range(7)],
                       "Y": [random.randint(0,5) for _ in range(7)],
                       "Z": [random.randint(0,6) for _ in range(7)],
                       "T": [random.randint(0,8) for _ in range(7)],
                       "R": [random.randint(0,2) for _ in range(7)],
                       "S": [random.randint(0,4) for _ in range(7)],
                       "P": [random.randint(0,5) for _ in range(7)],
                       "B": [random.randint(0,2) for _ in range(7)],
                       "N": [random.randint(0,2) for _ in range(7)]})

    print(TPDA_Algorithm(df))








