"""Reading the principal components: localization, sectors, eigenportfolios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inverse_participation_ratio(eigvecs: np.ndarray) -> np.ndarray:
    """IPR_k = sum_i v_ik^4.

    ~1/N for a vector spread evenly across all stocks (the market mode),
    ~3/N for a random vector, and up to 1 for a vector sitting on a single
    stock. Large IPR at the *edges* of the spectrum means localized modes.
    """
    return (eigvecs**4).sum(axis=0)


def sector_weights(eigvecs: np.ndarray, sectors: pd.Series, k: int = 10) -> pd.DataFrame:
    """Share of each PC's squared loadings that falls in each sector.

    Rows are PC1..PCk, columns sectors; each row sums to 1.
    """
    sq = pd.DataFrame(eigvecs[:, :k] ** 2, index=sectors.index)
    out = sq.groupby(sectors.values).sum().T
    out.index = [f"PC{i + 1}" for i in range(k)]
    return out


def sector_enrichment(eigvecs: np.ndarray, sectors: pd.Series, k: int = 10) -> pd.DataFrame:
    """sector_weights divided by the sector's share of stocks.

    1 = the PC loads on the sector exactly as much as a uniform vector would;
    5 = five times more. We return log2 so the scale is symmetric.
    """
    w = sector_weights(eigvecs, sectors, k)
    base = sectors.value_counts(normalize=True).reindex(w.columns)
    return np.log2(w / base)


def eigenportfolio_returns(returns: pd.DataFrame, eigvecs: np.ndarray, k: int = 5) -> pd.DataFrame:
    """Returns of the portfolios whose weights are the PC loadings / vol.

    Loadings live in standardized space, so we divide by each stock's vol to
    get dollar weights, then normalize to unit gross exposure.
    """
    vol = returns.std().values
    w = eigvecs[:, :k] / vol[:, None]
    w = w / np.abs(w).sum(axis=0)
    return pd.DataFrame(returns.values @ w, index=returns.index, columns=[f"PC{i + 1}" for i in range(k)])
