"""Does cleaning help? An out-of-sample minimum-variance test.

The global minimum-variance (GMV) portfolio w ~ Sigma^{-1} 1 is the purest
test of a covariance estimate: it has no expected-return input, and inverting
Sigma amplifies exactly the small, noisy eigenvalues that clipping fixes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .rmt import clip


def gmv_weights(cov: np.ndarray) -> np.ndarray:
    ones = np.ones(cov.shape[0])
    x = np.linalg.solve(cov, ones)
    return x / x.sum()


def rolling_gmv_backtest(
    returns: pd.DataFrame, train: int = 504, hold: int = 21
) -> pd.DataFrame:
    """Re-estimate every `hold` days on the trailing `train` days.

    Methods: sample correlation, RMT-clipped correlation (both with sample
    vols), and equal weight as a baseline. Returns daily out-of-sample returns.
    """
    x = returns.values
    n_days = len(returns)
    out = {"sample": [], "clipped": [], "equal": []}
    dates = []
    for t in range(train, n_days - 1, hold):
        window = x[t - train : t]
        vol = window.std(axis=0, ddof=1)
        corr = np.corrcoef(window, rowvar=False)
        cleaned, _ = clip(corr, n_obs=train)
        d = np.outer(vol, vol)
        w = {
            "sample": gmv_weights(corr * d),
            "clipped": gmv_weights(cleaned * d),
            "equal": np.full(x.shape[1], 1 / x.shape[1]),
        }
        future = x[t : t + hold]
        for k, wk in w.items():
            out[k].append(future @ wk)
        dates.append(returns.index[t : t + hold])
    idx = dates[0].append(dates[1:])
    return pd.DataFrame({k: np.concatenate(v) for k, v in out.items()}, index=idx)
