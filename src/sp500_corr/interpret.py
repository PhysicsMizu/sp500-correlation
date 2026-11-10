"""Tools for giving the signal PCs a meaning.

Five questions we ask of every component:
  1. Who loads on it?             -> top_loadings
  2. What does it trade like?     -> factor_correlations (vs. ETF proxies)
  3. How much does it explain?    -> r2_decomposition
  4. Is it there every year?      -> yearly_activity
  5. Is it a stable object?       -> eigvec_overlap / subspace_overlap
and one fix for when the answer to 5 is "no": varimax rotation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rmt import spectrum


def top_loadings(eigvecs: np.ndarray, meta: pd.DataFrame, k: int, n: int = 8) -> pd.DataFrame:
    """The n most positive and n most negative loadings of component k (0-based)."""
    v = pd.Series(eigvecs[:, k], index=meta.index, name="loading")
    idx = list(v.nlargest(n).index) + list(v.nsmallest(n).index[::-1])
    return meta.loc[idx, ["name", "sector", "sub_industry"]].assign(loading=v[idx])


def factor_correlations(pc_returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Correlation of each eigenportfolio's daily returns with each factor proxy."""
    j = pc_returns.join(factors, how="inner").dropna()
    return j.corr().loc[factors.columns, pc_returns.columns]


def r2_decomposition(eigvals: np.ndarray, eigvecs: np.ndarray, n_signal: int) -> pd.DataFrame:
    """Per stock: share of its variance explained by PC1 and by PCs 1..n_signal.

    The correlation of stock i with PC k is v_ik * sqrt(lambda_k), so its R^2
    on a set of (orthogonal) PCs is the sum of squared loadings times eigenvalues.
    """
    contrib = eigvecs**2 * eigvals
    return pd.DataFrame(
        {"market": contrib[:, 0], "signal": contrib[:, :n_signal].sum(axis=1)}
    )


def yearly_activity(returns: pd.DataFrame, eigvecs: np.ndarray, eigvals: np.ndarray, k: int) -> pd.DataFrame:
    """Variance of each PC's factor series per calendar year, relative to its full-sample average.

    1.0 = an ordinary year for that factor; 4 = the factor was 4x as loud.
    """
    z = (returns - returns.mean()) / returns.std()
    f = pd.DataFrame(z.values @ eigvecs[:, :k], index=returns.index,
                     columns=[f"PC{i + 1}" for i in range(k)])
    return f.groupby(f.index.year).var() / eigvals[:k]


def eigvec_overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """|<a_i, b_j>| for unit columns. 1 = same direction, ~sqrt(2/(pi N)) = random."""
    an = a / np.linalg.norm(a, axis=0)
    bn = b / np.linalg.norm(b, axis=0)
    return np.abs(an.T @ bn)


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of span(a) captured by span(b) (orthonormal columns). Random: k/N."""
    m = a.T @ b
    return float((m**2).sum() / a.shape[1])


def varimax(loadings: np.ndarray, max_iter: int = 500, tol: float = 1e-8) -> np.ndarray:
    """Kaiser's varimax rotation.

    Finds the orthogonal rotation R of the loading matrix that maximizes the
    variance of the squared loadings in each column, i.e. makes every factor
    load heavily on a few stocks and ~0 on the rest. The rotated factors span
    exactly the same subspace as the PCs, so nothing is lost; they just stop
    being forced to be ordered by variance.
    """
    p, k = loadings.shape

    def criterion(r):
        return ((loadings @ r) ** 2).var(axis=0).sum()

    rot = np.eye(k)
    best = criterion(rot)
    for _ in range(max_iter):
        lam = loadings @ rot
        u, _, vh = np.linalg.svd(
            loadings.T @ (lam**3 - lam @ np.diag((lam**2).sum(axis=0)) / p)
        )
        candidate = u @ vh
        value = criterion(candidate)
        # Keep only improving steps: on perfectly clean structure the
        # fixed-point update can oscillate between the optimum and a worse point
        if value <= best * (1 + tol):
            break
        rot, best = candidate, value
    out = loadings @ rot
    # Orient so the dominant loadings are positive
    out *= np.sign(out[np.abs(out).argmax(axis=0), np.arange(k)])
    return out


def rotated_factors(corr: np.ndarray, n_obs: int, k: int) -> np.ndarray:
    """Varimax-rotate PCs 2..k (the market mode is left alone: it's already clean)."""
    sp = spectrum(corr, n_obs)
    L = sp.eigvecs[:, 1:k] * np.sqrt(sp.eigvals[1:k])
    return varimax(L)


def name_factor(loading: pd.Series, meta: pd.DataFrame, n: int = 10) -> str:
    """Label a factor by the most common sub-industry among its top loadings."""
    top = loading.nlargest(n).index
    return meta.loc[top, "sub_industry"].value_counts().index[0]


def split_halves(returns: pd.DataFrame):
    h = len(returns) // 2
    return returns.iloc[:h], returns.iloc[h:]
