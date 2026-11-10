"""A small, honest backtest engine plus the statistics to judge its output.

Conventions (chosen to be conservative):
  * Weights for a rebalance on day t use returns up to and including t-1.
  * Trades happen at the close of day t; the new weights earn from t+1.
  * Between rebalances, weights drift with prices (buy and hold).
  * Turnover = sum |w_target - w_drifted| (cash included); cost = turnover * bps.
  * Cash (1 - sum w) earns the T-bill ETF return (BIL).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm

TRADING_DAYS = 252
WeightFn = Callable[[pd.DataFrame], np.ndarray]


# --- engine ---------------------------------------------------------------------------

@dataclass
class Result:
    returns: pd.Series      # daily net returns
    gross: pd.Series        # daily gross returns
    turnover: pd.Series     # one-way turnover on rebalance days
    weights: pd.DataFrame   # target weights on rebalance days (stocks only)


def month_starts(index: pd.DatetimeIndex, start: str, end: str) -> list[pd.Timestamp]:
    """First trading day of each month in [start, end]."""
    idx = index[(index >= start) & (index <= end)]
    s = pd.Series(idx, index=idx)
    return list(s.groupby([idx.year, idx.month]).first())


def run(
    returns: pd.DataFrame,
    cash: pd.Series,
    weight_fn: WeightFn,
    start: str,
    end: str,
    lookback: int,
    cost_bps: float = 5.0,
) -> Result:
    """Walk-forward backtest with monthly rebalancing.

    weight_fn receives the trailing `lookback` rows of `returns` ending the day
    *before* the rebalance and returns stock weights (sum <= 1; rest is cash).
    """
    idx = returns.index
    rebal = set(month_starts(idx, start, end))
    days = idx[(idx >= start) & (idx <= end)]
    pos = {d: i for i, d in enumerate(idx)}
    R = returns.values
    n = R.shape[1]

    w = np.zeros(n)
    w_cash = 1.0
    gross, net, tos, targets = [], [], {}, {}
    for d in days:
        i = pos[d]
        cost = 0.0
        if d in rebal:
            hist = returns.iloc[max(0, i - lookback):i]  # up to t-1
            target = np.asarray(weight_fn(hist), dtype=float)
            target_cash = 1.0 - target.sum()
            to = np.abs(target - w).sum() + abs(target_cash - w_cash)
            cost = to * cost_bps / 1e4
            tos[d] = to / 2  # one-way
            targets[d] = target
            w, w_cash = target, target_cash
        # earn the *next* day's return on the weights set today
        if i + 1 >= len(idx) or idx[i + 1] > pd.Timestamp(end):
            break
        r_next = R[i + 1]
        c_next = cash.iloc[i + 1]
        g = float(w @ r_next + w_cash * c_next)
        gross.append((idx[i + 1], g))
        net.append((idx[i + 1], g - cost))
        # drift
        grown = w * (1 + r_next)
        total = grown.sum() + w_cash * (1 + c_next)
        w, w_cash = grown / total, w_cash * (1 + c_next) / total

    g = pd.Series(dict(gross))
    return Result(
        returns=pd.Series(dict(net)),
        gross=g,
        turnover=pd.Series(tos),
        weights=pd.DataFrame(targets, index=returns.columns).T,
    )


# --- point statistics -------------------------------------------------------------------

def _moments(x: np.ndarray) -> tuple[float, float]:
    z = (x - x.mean()) / x.std()
    return float((z**3).mean()), float((z**4).mean())


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summary(r: pd.Series, rf: pd.Series, market: pd.Series | None = None) -> dict:
    """Annualized performance numbers for daily simple returns r."""
    rf = rf.reindex(r.index)
    ex = (r - rf).values
    T = len(r)
    sk, ku = _moments(r.values)
    monthly = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
    msk, _ = _moments(monthly.values)
    out = {
        "ann_return": float((1 + r).prod() ** (TRADING_DAYS / T) - 1),
        "ann_vol": float(r.std() * np.sqrt(TRADING_DAYS)),
        "sharpe": float(ex.mean() / ex.std() * np.sqrt(TRADING_DAYS)),
        "max_drawdown": max_drawdown(r),
        "skew_daily": sk,
        "skew_monthly": msk,
        "kurtosis_daily": ku,
        "cvar_5": float(np.sort(r.values)[: max(1, int(0.05 * T))].mean()),
        "years": T / TRADING_DAYS,
    }
    if market is not None:
        m = market.reindex(r.index)
        out["beta"] = float(np.cov(r, m)[0, 1] / m.var())
    return out


# --- inference for Sharpe ratios --------------------------------------------------------

def sharpe_se(ex: np.ndarray) -> float:
    """Std. error of the *per-period* Sharpe ratio under non-normal returns
    (Lo, 2002; Mertens, 2002): Var = (1 + SR^2/2 - skew*SR + (kurt-3)/4 * SR^2) / T."""
    sr = ex.mean() / ex.std()
    sk, ku = _moments(ex)
    return float(np.sqrt((1 + 0.5 * sr**2 - sk * sr + (ku - 3) / 4 * sr**2) / len(ex)))


def probabilistic_sharpe(ex: np.ndarray, sr_star: float = 0.0) -> float:
    """P(true per-period SR > sr_star) (Bailey & Lopez de Prado, 2012)."""
    sr = ex.mean() / ex.std()
    sk, ku = _moments(ex)
    denom = np.sqrt(1 - sk * sr + (ku - 1) / 4 * sr**2)
    return float(norm.cdf((sr - sr_star) * np.sqrt(len(ex) - 1) / denom))


def deflated_sharpe(ex: np.ndarray, trial_srs: np.ndarray) -> float:
    """Deflated Sharpe ratio (Bailey & Lopez de Prado, 2014).

    PSR against the Sharpe you'd expect from the *best of N* skill-less trials,
    given the dispersion of the N trial Sharpe ratios (all per-period).
    """
    n = len(trial_srs)
    gamma = 0.5772156649
    sd = np.std(trial_srs, ddof=1)
    sr_star = sd * ((1 - gamma) * norm.ppf(1 - 1 / n) + gamma * norm.ppf(1 - 1 / (n * np.e)))
    return probabilistic_sharpe(ex, sr_star)


def min_track_record_years(ex: np.ndarray, sr_star: float = 0.0, alpha: float = 0.05) -> float:
    """Years of data needed before the observed SR is significantly > sr_star."""
    sr = ex.mean() / ex.std()
    if sr <= sr_star:
        return float("inf")
    sk, ku = _moments(ex)
    n = 1 + (1 - sk * sr + (ku - 1) / 4 * sr**2) * (norm.ppf(1 - alpha) / (sr - sr_star)) ** 2
    return float(n / TRADING_DAYS)


# --- bootstrap ------------------------------------------------------------------------------

def stationary_bootstrap_indices(T: int, mean_block: int, rng: np.random.Generator) -> np.ndarray:
    """Politis & Romano (1994): blocks of geometric length, wrapping around."""
    idx = np.empty(T, dtype=int)
    i = rng.integers(T)
    p = 1.0 / mean_block
    for t in range(T):
        idx[t] = i
        i = rng.integers(T) if rng.random() < p else (i + 1) % T
    return idx


def bootstrap_metrics(
    rets: pd.DataFrame, rf: pd.Series, benchmark: str, n: int = 2000,
    mean_block: int = 21, seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Joint (paired) bootstrap of all strategies: CIs for Sharpe, vol, return,
    max drawdown, and each one's difference vs `benchmark`.

    Resampling whole days keeps the cross-sectional dependence between
    strategies, so the *differences* get honest (much tighter) intervals.
    """
    rng = np.random.default_rng(seed)
    X = rets.values
    ex = X - rf.reindex(rets.index).values[:, None]
    T, k = X.shape
    b = rets.columns.get_loc(benchmark)
    stats = {s: np.empty((n, k)) for s in ["sharpe", "vol", "ret", "mdd", "d_sharpe", "d_vol", "d_ret"]}
    for j in range(n):
        ii = stationary_bootstrap_indices(T, mean_block, rng)
        x, e = X[ii], ex[ii]
        sh = e.mean(0) / e.std(0) * np.sqrt(TRADING_DAYS)
        vol = x.std(0) * np.sqrt(TRADING_DAYS)
        ret = np.prod(1 + x, axis=0) ** (TRADING_DAYS / T) - 1
        wealth = np.cumprod(1 + x, axis=0)
        mdd = (wealth / np.maximum.accumulate(wealth, axis=0) - 1).min(0)
        stats["sharpe"][j], stats["vol"][j], stats["ret"][j], stats["mdd"][j] = sh, vol, ret, mdd
        stats["d_sharpe"][j] = sh - sh[b]
        stats["d_vol"][j] = vol - vol[b]
        stats["d_ret"][j] = ret - ret[b]
    out = {}
    for s, m in stats.items():
        df = pd.DataFrame(m, columns=rets.columns)
        out[s] = pd.DataFrame({
            "lo": df.quantile(0.025), "hi": df.quantile(0.975),
            # two-sided bootstrap p-value for "difference = 0"
            "p": 2 * np.minimum((df <= 0).mean(), (df >= 0).mean()),
        })
    return out


def holm(pvals: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values."""
    order = pvals.sort_values()
    m = len(order)
    adj = np.maximum.accumulate([(m - i) * p for i, p in enumerate(order.values)])
    return pd.Series(np.minimum(adj, 1), index=order.index).reindex(pvals.index)


# --- factor regressions -----------------------------------------------------------------------

def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """OLS with Newey-West (Bartlett) HAC standard errors. X should include a constant."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for l in range(1, lags + 1):
        w = 1 - l / (lags + 1)
        G = Xu[l:].T @ Xu[:-l]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    return beta, np.sqrt(np.diag(cov))


def factor_regression(ex: pd.Series, factors: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Daily excess returns on factor returns; alpha annualized (x252)."""
    j = pd.concat([ex.rename("y"), factors], axis=1, join="inner").dropna()
    X = np.column_stack([np.ones(len(j)), j[factors.columns].values])
    b, se = newey_west_ols(j["y"].values, X, lags)
    names = ["alpha (ann.)"] + list(factors.columns)
    scale = np.r_[TRADING_DAYS, np.ones(len(factors.columns))]
    resid = j["y"].values - X @ b
    r2 = 1 - resid.var() / j["y"].var()
    out = pd.DataFrame({"coef": b * scale, "t": b / se}, index=names)
    out.attrs["r2"] = float(r2)
    return out
