"""Portfolio construction rules. Each returns a function hist -> weights.

Every parameter here is fixed *before* looking at any backtest, using
textbook defaults. The training period is used to check that each mechanism
does what it claims, and the sensitivity grid is reported in full.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from .rmt import clip


def _ewma_vol(hist: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    x = hist.values
    w = lam ** np.arange(len(x))[::-1]
    w /= w.sum()
    return np.sqrt((w[:, None] * x**2).sum(axis=0))


def covariance(hist: pd.DataFrame, clean: bool = True) -> np.ndarray:
    """Correlation from the whole window (optionally RMT-clipped) x EWMA vols."""
    c = np.corrcoef(hist.values, rowvar=False)
    if clean:
        c, _ = clip(c, n_obs=len(hist))
    vol = _ewma_vol(hist)
    return c * np.outer(vol, vol)


# --- 4. the market ---------------------------------------------------------------------

def equal_weight():
    def f(hist: pd.DataFrame) -> np.ndarray:
        n = hist.shape[1]
        return np.full(n, 1 / n)
    return f


# --- 1. low volatility -----------------------------------------------------------------

def min_variance(cap: float = 0.05, clean: bool = True):
    """Long-only global minimum variance, w_i in [0, cap], fully invested.

    No expected returns anywhere: they can't be estimated with any useful
    precision. Any extra return has to come from the low-volatility anomaly.
    """
    def f(hist: pd.DataFrame) -> np.ndarray:
        return solve_min_variance(covariance(hist, clean=clean), cap)
    return f


def solve_min_variance(S: np.ndarray, cap: float) -> np.ndarray:
    """Long-only, fully invested, w_i <= cap, minimum w'Sw."""
    import cvxpy as cp

    S = (S + S.T) / 2
    n = len(S)
    # Square-root factor: w'Sw = ||L'w||^2. Eigen-based so it also works
    # when S is singular (raw sample covariance with N > T).
    vals, vecs = np.linalg.eigh(S)
    L = vecs * np.sqrt(np.clip(vals, 0, None))
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(L.T @ w)),
                      [cp.sum(w) == 1, w >= 0, w <= cap])
    prob.solve(solver=cp.CLARABEL)
    x = np.clip(np.asarray(w.value).ravel(), 0, None)
    return x / x.sum()


def regime_min_variance(probs: pd.DataFrame, A: np.ndarray, cap: float = 0.05, horizon: int = 21):
    """Minimum variance on a covariance that mixes regime-specific covariances.

    probs: *filtered* (causal) HMM state probabilities of the market, by date.
    For each regime k, every day in the trailing window is weighted by
    P(s_day = k), giving a regime correlation (RMT-clipped at its effective
    sample size) and regime vols. The forecast covariance for the coming month
    mixes them with the average predicted state probabilities over the next
    `horizon` days: Sigma = sum_k pbar_k Sigma_k.
    """
    def f(hist: pd.DataFrame) -> np.ndarray:
        p = probs.reindex(hist.index).ffill().fillna(1 / probs.shape[1]).values
        x = hist.values - hist.values.mean(axis=0)
        S = np.zeros((x.shape[1], x.shape[1]))
        p_now = p[-1]
        pbar = np.mean([p_now @ np.linalg.matrix_power(A, h) for h in range(1, horizon + 1)], axis=0)
        for k in range(p.shape[1]):
            w = p[:, k]
            n_eff = w.sum() ** 2 / (w**2).sum()
            cov = (x * w[:, None]).T @ x / w.sum()
            sd = np.sqrt(np.diag(cov))
            c = cov / np.outer(sd, sd)
            if n_eff > 30:
                c, _ = clip(c, n_obs=int(n_eff))
            S += pbar[k] * c * np.outer(sd, sd)
        return solve_min_variance(S, cap)
    return f


# --- 2. skew -----------------------------------------------------------------------------

def trend_filter(sma_days: int = 210):
    """Hold the equal-weight universe when it's above its ~10-month moving
    average, otherwise T-bills (Faber, 2007).

    Why this creates positive skew: a trend rule is approximately a
    replicating strategy for a straddle. It adds exposure after rises and
    cuts it after falls, so it's convex in the market. Part II showed that
    the market mode is where the negative skew lives, so that's where to
    attack it.
    """
    def f(hist: pd.DataFrame) -> np.ndarray:
        n = hist.shape[1]
        idx = (1 + hist.mean(axis=1)).cumprod()
        if len(idx) < sma_days:
            return np.full(n, 1 / n)
        on = idx.iloc[-1] > idx.iloc[-sma_days:].mean()
        return np.full(n, 1 / n) if on else np.zeros(n)
    return f


def lottery(top: int = 50, window: int = 252):
    """The naive way to 'buy skew': equal-weight the stocks with the highest
    trailing skewness. A foil for the trend filter: portfolio skew is mostly
    *co*-skewness (Part II), and diversification erases idiosyncratic skew."""
    def f(hist: pd.DataFrame) -> np.ndarray:
        x = hist.iloc[-window:]
        z = (x - x.mean()) / x.std()
        sk = (z**3).mean().values
        w = np.zeros(hist.shape[1])
        w[np.argsort(-sk)[:top]] = 1 / top
        return w
    return f


# --- 3. hierarchical risk parity ----------------------------------------------------------

def _hrp_bisect(cov: np.ndarray, order: list[int]) -> np.ndarray:
    w = np.ones(len(cov))
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            a, b = c[: len(c) // 2], c[len(c) // 2:]

            def cvar(ix):
                sub = cov[np.ix_(ix, ix)]
                ivp = 1 / np.diag(sub)
                ivp /= ivp.sum()
                return ivp @ sub @ ivp

            va, vb = cvar(a), cvar(b)
            alpha = 1 - va / (va + vb)
            w[a] *= alpha
            w[b] *= 1 - alpha
            nxt += [a, b]
        clusters = nxt
    return w / w.sum()


def hrp(clean: bool = False, method: str = "single"):
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    1. Tree: single-linkage clustering on d = sqrt((1 - rho)/2). Single
       linkage *is* the minimum spanning tree from the README.
    2. Quasi-diagonalize: order stocks so neighbours in the tree are adjacent.
    3. Recursive bisection: split the ordered list in halves, give each half
       weight inversely proportional to its variance.
    No matrix inversion, so no amplification of noisy small eigenvalues.
    """
    def f(hist: pd.DataFrame) -> np.ndarray:
        S = covariance(hist, clean=clean)
        sd = np.sqrt(np.diag(S))
        corr = np.clip(S / np.outer(sd, sd), -1, 1)
        dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
        np.fill_diagonal(dist, 0)
        link = linkage(squareform(dist, checks=False), method=method)
        order = list(leaves_list(link))
        return _hrp_bisect(S, order)
    return f


# --- simple single-asset trend following -------------------------------------------------

def single_asset_trend(
    price: pd.Series,
    risky: pd.Series,
    cash: pd.Series,
    start: str,
    end: str,
    sma_days: int = 200,
    check: str = "daily",
    cost_bps: float = 5.0,
) -> tuple[pd.Series, pd.Series]:
    """Long `risky` when `price` is above its `sma_days` moving average, else cash.

    Timing (same conservative convention as the portfolio engine): the signal
    computed from the close of day t-1 is traded at the close of day t and
    earns from day t+1. A full switch costs 2 x cost_bps (sell one, buy the
    other). Returns (daily net returns, position held each day).
    """
    above = (price > price.rolling(sma_days).mean()).astype(float)
    if check == "monthly":
        # only allowed to change on the first trading day of each month
        first = price.index.to_series().groupby([price.index.year, price.index.month]).transform("first")
        above = above.where(price.index == first).ffill()
    # signal known at close t-1 -> traded at close t -> earns at t+1
    pos = above.shift(2)
    days = price.index[(price.index >= start) & (price.index <= end)]
    pos = pos.reindex(days).fillna(0.0)
    trades = pos.diff().abs().fillna(pos.iloc[0])
    r = pos * risky.reindex(days) + (1 - pos) * cash.reindex(days) - trades * 2 * cost_bps / 1e4
    return r, pos


def min_variance_window(cap: float = 0.05):
    """Ablation baseline: RMT-clipped correlation x *window* (not EWMA) vols, no regimes."""
    def f(hist: pd.DataFrame) -> np.ndarray:
        c, _ = clip(np.corrcoef(hist.values, rowvar=False), n_obs=len(hist))
        sd = hist.std().values
        return solve_min_variance(c * np.outer(sd, sd), cap)
    return f
