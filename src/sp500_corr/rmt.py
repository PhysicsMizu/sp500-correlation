"""Random matrix theory tools: Marchenko-Pastur law and eigenvalue clipping.

If you have T observations of N *independent* unit-variance series, the
eigenvalues of their sample correlation matrix don't all equal 1: finite-sample
noise spreads them over the Marchenko-Pastur (MP) interval

    lambda_{+/-} = sigma^2 (1 +/- sqrt(q))^2,     q = N / T.

Eigenvalues of a real correlation matrix that sit inside that interval are
indistinguishable from noise. "Clipping" (Laloux, Cizeau, Bouchaud & Potters,
1999) keeps the eigenvalues above lambda_+ and flattens the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mp_bounds(q: float, sigma2: float = 1.0) -> tuple[float, float]:
    """Edges of the Marchenko-Pastur bulk for aspect ratio q = N/T."""
    s = np.sqrt(q)
    return sigma2 * (1 - s) ** 2, sigma2 * (1 + s) ** 2


def mp_pdf(x: np.ndarray, q: float, sigma2: float = 1.0) -> np.ndarray:
    """Marchenko-Pastur density (for q <= 1)."""
    lo, hi = mp_bounds(q, sigma2)
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    inside = (x > lo) & (x < hi)
    xi = x[inside]
    out[inside] = np.sqrt((hi - xi) * (xi - lo)) / (2 * np.pi * q * sigma2 * xi)
    return out


def fit_noise_variance(eigvals: np.ndarray) -> float:
    """Variance of the 'noise' part of the spectrum (Laloux et al., 1999).

    The trace of a correlation matrix is N. The market mode alone carries
    lambda_1 of it, so whatever is left to be "noise" has variance
    sigma^2 = 1 - lambda_1 / N, which narrows the MP bulk.

    (Tempting refinements, like iterating "sigma^2 = mean of the bulk", are
    unstable here: every shrink of lambda_+ evicts more eigenvalues from the
    bulk and the fit collapses to zero. Try it!)
    """
    n = len(eigvals)
    return float(1.0 - eigvals.max() / n)


@dataclass
class Spectrum:
    eigvals: np.ndarray  # descending
    eigvecs: np.ndarray  # columns, matching eigvals
    q: float
    sigma2: float
    lambda_minus: float
    lambda_plus: float

    @property
    def n_signal(self) -> int:
        return int((self.eigvals > self.lambda_plus).sum())


def spectrum(corr: np.ndarray, n_obs: int, fit_sigma: bool = True) -> Spectrum:
    """Eigendecomposition (descending) plus the MP noise band."""
    n = corr.shape[0]
    q = n / n_obs
    vals, vecs = np.linalg.eigh(corr)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    # Eigenvectors are defined up to sign; pick the one whose loadings sum positive
    signs = np.sign(vecs.sum(axis=0))
    signs[signs == 0] = 1
    vecs = vecs * signs
    sigma2 = fit_noise_variance(vals) if fit_sigma else 1.0
    lo, hi = mp_bounds(q, sigma2)
    return Spectrum(vals, vecs, q, sigma2, lo, hi)


def _to_correlation(m: np.ndarray) -> np.ndarray:
    d = 1.0 / np.sqrt(np.diag(m))
    c = m * np.outer(d, d)
    np.fill_diagonal(c, 1.0)
    return c


def clip(corr: np.ndarray, n_obs: int, fit_sigma: bool = True) -> tuple[np.ndarray, Spectrum]:
    """Eigenvalue clipping.

    Eigenvalues inside the MP bulk are replaced by their average, which
    preserves the trace; we then rescale back to a unit diagonal.
    """
    spec = spectrum(corr, n_obs, fit_sigma=fit_sigma)
    vals = spec.eigvals.copy()
    noise = vals <= spec.lambda_plus
    if noise.any():
        vals[noise] = vals[noise].mean()
    cleaned = (spec.eigvecs * vals) @ spec.eigvecs.T
    return _to_correlation(cleaned), spec


def detone(corr: np.ndarray, n_market: int = 1) -> np.ndarray:
    """Remove the top `n_market` modes (the market) to expose sector structure.

    The result is singular, so use it for clustering/trees, not for inversion.
    """
    vals, vecs = np.linalg.eigh(corr)
    vals, vecs = vals[::-1], vecs[:, ::-1]
    market = (vecs[:, :n_market] * vals[:n_market]) @ vecs[:, :n_market].T
    return _to_correlation(corr - market)
