"""Markov chains for market regimes: observable chains, tests, and a Gaussian HMM.

Two kinds of chain:

* Observable: you define the states (e.g. "this month was down and volatile")
  and estimate P(next state | current state) by counting transitions.
* Hidden (HMM): the states are latent; each emits returns from its own
  normal distribution N(mu_k, sigma_k^2). We only see the returns.

The single most important distinction for a backtest:

    filtered  P(s_t | r_1..r_t)  -- uses data up to today      -> tradable
    smoothed  P(s_t | r_1..r_T)  -- uses the *whole* sample    -> look-ahead

Smoothed probabilities are what most HMM tutorials plot, and what most
"regime" backtests accidentally trade on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm


# --- observable chains ---------------------------------------------------------------

def transition_counts(states: np.ndarray, k: int) -> np.ndarray:
    n = np.zeros((k, k))
    np.add.at(n, (states[:-1], states[1:]), 1)
    return n


def transition_matrix(states: np.ndarray, k: int) -> np.ndarray:
    n = transition_counts(states, k)
    return n / n.sum(axis=1, keepdims=True)


def stationary_distribution(P: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(P.T)
    v = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return v / v.sum()


def expected_durations(P: np.ndarray) -> np.ndarray:
    """Mean time spent in a state before leaving: 1 / (1 - p_ii)."""
    return 1 / (1 - np.diag(P))


def markov_order_test(states: np.ndarray, k: int, order: int = 1) -> dict:
    """Likelihood-ratio test (Anderson & Goodman, 1957).

    order=1: H0 "states are i.i.d."            vs H1 "first-order Markov"
    order=2: H0 "first-order Markov"            vs H1 "second-order Markov"
    """
    s = np.asarray(states)
    if order == 1:
        n = transition_counts(s, k)
        P = n / n.sum(axis=1, keepdims=True)
        p = n.sum(axis=0) / n.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = 2 * np.nansum(n * np.log(P / p[None, :]))
        df = (k - 1) ** 2
    else:
        n3 = np.zeros((k, k, k))
        np.add.at(n3, (s[:-2], s[1:-1], s[2:]), 1)
        P2 = n3 / n3.sum(axis=2, keepdims=True)
        n2 = n3.sum(axis=0)
        P1 = n2 / n2.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            lr = 2 * np.nansum(n3 * np.log(P2 / P1[None, :, :]))
        df = k * (k - 1) ** 2
    return {"lr": float(lr), "df": df, "p": float(chi2.sf(lr, df))}


# --- Gaussian hidden Markov model -----------------------------------------------------

@dataclass
class GaussianHMM:
    pi: np.ndarray      # initial state distribution (k,)
    A: np.ndarray       # transition matrix (k, k)
    mu: np.ndarray      # emission means (k,)
    sigma: np.ndarray   # emission std devs (k,)
    loglik: float = np.nan

    @property
    def k(self) -> int:
        return len(self.mu)

    def n_params(self) -> int:
        k = self.k
        return (k - 1) + k * (k - 1) + 2 * k

    def bic(self, n_obs: int) -> float:
        return -2 * self.loglik + self.n_params() * np.log(n_obs)

    def _emission(self, x: np.ndarray) -> np.ndarray:
        return norm.pdf(x[:, None], self.mu[None, :], self.sigma[None, :]) + 1e-300

    def filter(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """Forward algorithm with scaling. Row t = P(s_t | x_1..x_t). Causal."""
        B = self._emission(x)
        T, k = B.shape
        alpha = np.empty((T, k))
        ll = 0.0
        a = self.pi * B[0]
        for t in range(T):
            if t > 0:
                a = (alpha[t - 1] @ self.A) * B[t]
            c = a.sum()
            alpha[t] = a / c
            ll += np.log(c)
        return alpha, ll

    def smooth(self, x: np.ndarray) -> np.ndarray:
        """Forward-backward. Row t = P(s_t | x_1..x_T). NOT causal: uses the future."""
        alpha, _ = self.filter(x)
        B = self._emission(x)
        T, k = B.shape
        beta = np.ones((T, k))
        for t in range(T - 2, -1, -1):
            b = self.A @ (B[t + 1] * beta[t + 1])
            beta[t] = b / b.sum()
        g = alpha * beta
        return g / g.sum(axis=1, keepdims=True)

    def predict(self, filtered: np.ndarray, steps: int = 1) -> np.ndarray:
        """P(s_{t+steps} | x_1..x_t) from filtered probabilities."""
        return filtered @ np.linalg.matrix_power(self.A, steps)


def _sort_states(m: GaussianHMM) -> GaussianHMM:
    """Order states from calmest to most volatile, so 'state 0' means the same thing every fit."""
    o = np.argsort(m.sigma)
    return GaussianHMM(m.pi[o], m.A[np.ix_(o, o)], m.mu[o], m.sigma[o], m.loglik)


def fit_hmm(x: np.ndarray, k: int, n_init: int = 5, max_iter: int = 300, tol: float = 1e-7,
            seed: int = 0) -> GaussianHMM:
    """Baum-Welch (EM) with several random starts; keeps the best likelihood."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    T = len(x)
    best = None
    for _ in range(n_init):
        # init: quantile-spread volatilities, sticky transitions
        sd = x.std() * np.exp(rng.uniform(-0.8, 0.8, k))
        m = GaussianHMM(np.full(k, 1 / k),
                        np.full((k, k), 0.02 / max(k - 1, 1)) + np.eye(k) * (0.98 - 0.02 / max(k - 1, 1)),
                        np.full(k, x.mean()) + rng.normal(0, x.std() * 0.05, k), sd)
        prev = -np.inf
        for _ in range(max_iter):
            alpha, ll = m.filter(x)
            B = m._emission(x)
            beta = np.ones((T, k))
            for t in range(T - 2, -1, -1):
                b = m.A @ (B[t + 1] * beta[t + 1])
                beta[t] = b / b.sum()
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True)
            xi = alpha[:-1, :, None] * m.A[None] * (B[1:] * beta[1:])[:, None, :]
            xi /= xi.sum(axis=(1, 2), keepdims=True)
            w = gamma.sum(axis=0)
            mu = (gamma * x[:, None]).sum(axis=0) / w
            sigma = np.sqrt((gamma * (x[:, None] - mu) ** 2).sum(axis=0) / w)
            A = xi.sum(axis=0)
            A /= A.sum(axis=1, keepdims=True)
            m = GaussianHMM(gamma[0], A, mu, np.maximum(sigma, 1e-6), ll)
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        _, ll = m.filter(x)
        m.loglik = ll
        if best is None or ll > best.loglik:
            best = m
    return _sort_states(best)


# --- forecasts and exposures built on an HMM ---------------------------------------------

def hmm_variance_forecast(m: GaussianHMM, filtered: np.ndarray) -> np.ndarray:
    """One-step-ahead variance of the mixture: E[r^2] - E[r]^2 under P(s_{t+1} | x_1..x_t)."""
    p = m.predict(filtered, 1)
    e1 = p @ m.mu
    e2 = p @ (m.sigma**2 + m.mu**2)
    return e2 - e1**2


def ewma_variance_forecast(x: np.ndarray, lam: float = 0.94, warmup: int = 60) -> np.ndarray:
    """RiskMetrics: forecast for t+1 made at t = lam * forecast_t + (1 - lam) * x_t^2."""
    v = np.empty(len(x))
    v[0] = np.var(x[:warmup])
    for t in range(1, len(x)):
        v[t] = lam * v[t - 1] + (1 - lam) * x[t - 1] ** 2
    # v[t] is the forecast for day t made at t-1; shift so row t = forecast for t+1 made at t
    return np.r_[v[1:], lam * v[-1] + (1 - lam) * x[-1] ** 2]


def qlike(realized_sq: np.ndarray, forecast_var: np.ndarray) -> np.ndarray:
    """QLIKE loss (Patton, 2011): robust to noise in the realized-variance proxy."""
    return realized_sq / forecast_var + np.log(forecast_var)


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lags: int = 10) -> dict:
    """Test equal predictive accuracy; negative stat => a has lower loss. Newey-West variance."""
    d = loss_a - loss_b
    T = len(d)
    dc = d - d.mean()
    s = dc @ dc / T
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (dc[l:] @ dc[:-l]) / T
    stat = d.mean() / np.sqrt(s / T)
    return {"mean_diff": float(d.mean()), "stat": float(stat), "p": float(2 * norm.sf(abs(stat)))}


def walk_forward_hmm(x: "pd.Series", first_year: int, ks=(2, 3, 4), n_init: int = 4) -> dict:
    """Refit every January on all data *before* that year (K chosen by BIC each time).

    Returns, for each date from `first_year` on: filtered probabilities of the
    'risk-off' states (negative fitted mean), the one-step variance forecast,
    and which K was chosen.
    """
    import pandas as pd

    out_p, out_v, out_k = [], [], {}
    years = sorted(set(x.index.year))
    for y in [yy for yy in years if yy >= first_year]:
        train = x[x.index.year < y].values
        fits = [fit_hmm(train, k, n_init=n_init, seed=y) for k in ks]
        m = min(fits, key=lambda f: f.bic(len(train)))
        out_k[y] = m.k
        upto = x[x.index.year <= y]
        filt, _ = m.filter(upto.values)
        pred = m.predict(filt, 1)
        mask = upto.index.year == y
        off = m.mu < 0
        out_p.append(pd.Series(pred[mask][:, off].sum(axis=1), index=upto.index[mask]))
        out_v.append(pd.Series(hmm_variance_forecast(m, filt)[mask], index=upto.index[mask]))
    return {"p_off": pd.concat(out_p), "var": pd.concat(out_v), "k": out_k}
