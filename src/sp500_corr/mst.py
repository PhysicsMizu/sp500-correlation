"""Minimum spanning trees of the correlation network (Mantegna, 1999).

Correlation isn't a distance, but d_ij = sqrt(2 (1 - rho_ij)) is: it's the
Euclidean distance between the standardized return vectors (scaled by 1/sqrt(T)).
The MST keeps the N-1 shortest links that connect every stock without loops,
a sparse skeleton of the strongest co-movements.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree


def corr_to_distance(corr: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))


def _mst_edges(corr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dist = corr_to_distance(corr)
    np.fill_diagonal(dist, 0.0)  # scipy treats 0 as "no edge"
    tree = minimum_spanning_tree(dist).tocoo()
    return tree.row, tree.col, tree.data


def build_mst(corr: np.ndarray, labels: list[str]) -> nx.Graph:
    rows, cols, d = _mst_edges(corr)
    g = nx.Graph()
    g.add_nodes_from(labels)
    for i, j, dij in zip(rows, cols, d):
        g.add_edge(labels[i], labels[j], weight=float(dij), corr=float(corr[i, j]))
    return g


def normalized_tree_length(corr: np.ndarray) -> float:
    """Mean edge distance of the MST (Onnela et al., 2003).

    It shrinks when markets crash, as everything starts moving together.
    """
    _, _, d = _mst_edges(corr)
    return float(d.mean())


def sector_purity(g: nx.Graph, sectors: pd.Series, n_perm: int = 2000, seed: int = 0) -> dict:
    """Share of MST edges that link two stocks of the same sector, vs a shuffle null."""
    edges = list(g.edges())
    nodes = list(g.nodes())
    idx = {n: k for k, n in enumerate(nodes)}
    a = np.array([idx[u] for u, _ in edges])
    b = np.array([idx[v] for _, v in edges])
    lab = sectors.reindex(nodes).values
    observed = float((lab[a] == lab[b]).mean())
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        p = rng.permutation(lab)
        null[k] = (p[a] == p[b]).mean()
    return {
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "p_value": float((1 + (null >= observed).sum()) / (1 + n_perm)),
    }


def hubs(g: nx.Graph, top: int = 10) -> pd.Series:
    return pd.Series(dict(g.degree())).sort_values(ascending=False).head(top)


def rolling_stats(returns: pd.DataFrame, window: int = 252, step: int = 21) -> pd.DataFrame:
    """Over rolling windows: top eigenvalue share and normalized tree length."""
    x = returns.values
    n = x.shape[1]
    rows = []
    for t in range(window, len(returns) + 1, step):
        c = np.corrcoef(x[t - window : t], rowvar=False)
        lam1 = np.linalg.eigvalsh(c)[-1]
        off = c[~np.eye(n, dtype=bool)]
        rows.append(
            {
                "date": returns.index[t - 1],
                "lambda1_share": lam1 / n,
                "mean_corr": off.mean(),
                "tree_length": normalized_tree_length(c),
            }
        )
    return pd.DataFrame(rows).set_index("date")
