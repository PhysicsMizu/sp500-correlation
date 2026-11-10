"""Beyond second moments: do stocks fall together more than they rise together?

Correlation is blind to direction: flip the sign of every return and C does
not change. Everything here is a statistic that *does* change under that
flip, which gives a natural null model:

    sign-flip null: multiply all returns on day t by the same random s_t = +-1.

That keeps the correlation matrix, fat tails and volatility clustering intact
but makes "down" and "up" statistically identical. Any down-vs-up difference
larger than the null spread is genuine asymmetry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import mst, rmt


# --- preprocessing --------------------------------------------------------------

def ewma_devol(returns: pd.DataFrame, lam: float = 0.94, warmup: int = 60) -> pd.DataFrame:
    """Divide each return by a RiskMetrics EWMA volatility built from *past* returns only.

    Crashes cluster in high-volatility regimes. Without this step a handful of
    days (March 2020) dominate every tail statistic and nothing is testable.
    """
    # No demeaning here: a full-sample mean would leak the future into the past.
    # (RiskMetrics assumes a zero daily mean; the final standardization is an
    # affine map per column and doesn't affect any correlation-type statistic.)
    x = returns.values
    s2 = np.empty_like(x)
    s2[0] = x[:warmup].var(axis=0)
    for t in range(1, len(x)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * x[t - 1] ** 2
    out = pd.DataFrame(x / np.sqrt(s2), index=returns.index, columns=returns.columns).iloc[warmup:]
    return (out - out.mean()) / out.std()


def standardize(returns: pd.DataFrame) -> pd.DataFrame:
    return (returns - returns.mean()) / returns.std()


def sign_flip(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return y * rng.choice([-1.0, 1.0], size=(len(y), 1))


def _off(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)


# --- 1. conditional (down vs up market) correlation ------------------------------

def conditional_corr(y: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Correlation on days when the equal-weight market is in its bottom / top q."""
    m = y.mean(axis=1)
    lo, hi = np.quantile(m, [q, 1 - q])
    down, up = m <= lo, m >= hi
    return np.corrcoef(y[down], rowvar=False), np.corrcoef(y[up], rowvar=False), int(down.sum())


def conditional_stats(y: np.ndarray, q: float = 0.2) -> dict:
    c_dn, c_up, n_days = conditional_corr(y, q)
    off = _off(y.shape[1])
    out = {}
    for name, c in [("down", c_dn), ("up", c_up)]:
        spec = rmt.spectrum(c, n_days)
        out[name] = {
            "mean_corr": float(c[off].mean()),
            "lambda1_share": float(spec.eigvals[0] / len(c)),
            "lambda2": float(spec.eigvals[1]),
            "n_signal": spec.n_signal,
            "tree_length": mst.normalized_tree_length(c),
        }
    return out


def conditional_null(y: np.ndarray, q: float, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sign-flip null distribution of (down - up) for each conditional statistic."""
    rows = []
    for _ in range(n):
        s = conditional_stats(sign_flip(y, rng), q)
        rows.append({k: s["down"][k] - s["up"][k] for k in s["down"]})
    return pd.DataFrame(rows)


def exceedance_curve(y: np.ndarray, qs: np.ndarray) -> pd.DataFrame:
    """Mean pairwise correlation on bottom-q and top-q market days, for each q."""
    off = _off(y.shape[1])
    rows = []
    for q in qs:
        c_dn, c_up, _ = conditional_corr(y, q)
        rows.append({"q": q, "down": c_dn[off].mean(), "up": c_up[off].mean()})
    return pd.DataFrame(rows).set_index("q")


def gaussian_twin(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multivariate normal sample with the same correlation matrix and length."""
    c = np.corrcoef(y, rowvar=False)
    chol = np.linalg.cholesky(c + 1e-8 * np.eye(len(c)))
    return rng.standard_normal(y.shape) @ chol.T


# --- 2. semicovariance -------------------------------------------------------------

def semicovariance(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact split C = P + N + M (Bollerslev, Li, Patton & Quaedvlieg, 2020).

    P: both returns positive, N: both negative, M: mixed signs (<= 0).
    A sign flip of a day moves its contribution from P to N and back, so
    under the null E[N] = E[P].
    """
    pos, neg = np.maximum(y, 0), np.minimum(y, 0)
    t = len(y)
    return pos.T @ pos / t, neg.T @ neg / t, (pos.T @ neg + neg.T @ pos) / t


def downside_excess(y: np.ndarray) -> np.ndarray:
    """A = N - P: where joint down-moves exceed joint up-moves."""
    p, n, _ = semicovariance(y)
    return n - p


# --- 3. tail dependence ------------------------------------------------------------

def to_uniform(y: np.ndarray) -> np.ndarray:
    """Probability integral transform via ranks (removes all marginal effects)."""
    return rankdata(y, axis=0) / (len(y) + 1)


def tail_dependence(u: np.ndarray, q: float) -> tuple[np.ndarray, np.ndarray]:
    """lambda_L(q)_ij = P(U_i <= q | U_j <= q) and the same for the upper tail."""
    lo = (u <= q).astype(float)
    hi = (u >= 1 - q).astype(float)
    t = len(u)
    return lo.T @ lo / (t * q), hi.T @ hi / (t * q)


def gaussian_copula_twin(u: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniforms from a Gaussian copula with the data's normal-scores correlation."""
    from scipy.stats import norm

    c = np.corrcoef(norm.ppf(u), rowvar=False)
    chol = np.linalg.cholesky(c + 1e-8 * np.eye(len(c)))
    return to_uniform(rng.standard_normal(u.shape) @ chol.T)


def sector_block_means(a: np.ndarray, sectors: np.ndarray) -> pd.DataFrame:
    """Average of a pairwise matrix within and between sectors (diagonal excluded)."""
    names = np.unique(sectors)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for s1 in names:
        for s2 in names:
            block = a[np.ix_(sectors == s1, sectors == s2)]
            if s1 == s2:
                block = block[_off(len(block))]
            out.loc[s1, s2] = block.mean()
    return out


# --- 4. coskewness tensor ----------------------------------------------------------

def whiten(y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project onto the top-k PCs and scale to unit variance: F = Z V / sqrt(lambda)."""
    c = np.corrcoef(y, rowvar=False)
    vals, vecs = np.linalg.eigh(c)
    vals, vecs = vals[::-1][:k], vecs[:, ::-1][:, :k]
    vecs = vecs * np.sign(vecs.sum(axis=0))
    z = (y - y.mean(axis=0)) / y.std(axis=0)
    return z @ vecs / np.sqrt(vals), vecs, vals


def coskewness(f: np.ndarray) -> np.ndarray:
    """S_abc = E[f_a f_b f_c] for whitened factors (a k x k x k symmetric tensor).

    For all N stocks this tensor would have N^3 ~ 95 million entries; in the
    whitened top-k subspace it has k^3 (1,728 for k = 12).
    """
    return np.einsum("ta,tb,tc->abc", f, f, f) / len(f)


def tensor_power_decomposition(
    s: np.ndarray, n_comp: int, rng: np.random.Generator, n_init: int = 30, n_iter: int = 200
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy symmetric CP decomposition S ~ sum_r kappa_r u_r (x) u_r (x) u_r.

    Tensor power method with random restarts and deflation (Anandkumar et al.,
    2014). The fixed points u = S(., u, u) / ||S(., u, u)|| are the tensor's
    Z-eigenvectors: the directions of locally maximal skewness. If returns were
    generated by independent, skewed factors, these directions would identify
    the factors *uniquely*. That's the theorem behind ICA, and it's what second
    moments can't do (they leave any rotation free).

    Sign convention: each u_r is oriented so its skewness kappa_r is positive.
    """
    s = s.copy()
    k = s.shape[0]
    kappas, us = [], []
    for _ in range(n_comp):
        best_val, best_u = 0.0, None
        for _ in range(n_init):
            u = rng.standard_normal(k)
            u /= np.linalg.norm(u)
            for _ in range(n_iter):
                v = np.einsum("abc,b,c->a", s, u, u)
                nv = np.linalg.norm(v)
                if nv < 1e-12:
                    break
                u = v / nv
            val = np.einsum("abc,a,b,c->", s, u, u, u)
            if best_u is None or abs(val) > abs(best_val):
                best_val, best_u = val, u
        if best_val < 0:
            best_val, best_u = -best_val, -best_u
        kappas.append(best_val)
        us.append(best_u)
        s = s - best_val * np.einsum("a,b,c->abc", best_u, best_u, best_u)
    return np.array(kappas), np.array(us).T


def block_bootstrap_skew(x: np.ndarray, rng: np.random.Generator, n: int = 500,
                         block: int = 20) -> np.ndarray:
    """Moving-block bootstrap distribution of the skewness of a series."""
    t = len(x)
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, t - block, t // block)
        xb = x[(starts[:, None] + np.arange(block)).ravel()]
        xb = (xb - xb.mean()) / xb.std()
        out[i] = (xb**3).mean()
    return out


def skewness(x: np.ndarray) -> float:
    x = (x - x.mean()) / x.std()
    return float((x**3).mean())
