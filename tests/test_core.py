import networkx as nx
import numpy as np
import pandas as pd

from sp500_corr import interpret, mst, pca, rmt


def _corr(x):
    return np.corrcoef(x, rowvar=False)


def test_pure_noise_stays_inside_mp_bulk():
    rng = np.random.default_rng(0)
    T, N = 2000, 200
    c = _corr(rng.standard_normal((T, N)))
    spec = rmt.spectrum(c, T, fit_sigma=False)
    lo, hi = rmt.mp_bounds(N / T)
    # Finite-size edge fluctuations are O(N^{-2/3}); allow a small margin
    assert spec.eigvals.max() < hi * 1.05
    assert spec.eigvals.min() > lo * 0.9


def test_mp_pdf_integrates_to_one():
    q = 0.3
    lo, hi = rmt.mp_bounds(q)
    x = np.linspace(lo, hi, 200_001)
    assert abs(np.trapezoid(rmt.mp_pdf(x, q), x) - 1) < 1e-3


def test_planted_factor_is_detected_and_clipping_is_valid():
    rng = np.random.default_rng(1)
    T, N = 1000, 150
    market = rng.standard_normal((T, 1))
    x = 0.5 * market + rng.standard_normal((T, N))
    c = _corr(x)
    clean, spec = rmt.clip(c, T)
    assert spec.n_signal == 1
    assert np.allclose(np.diag(clean), 1)
    assert np.allclose(clean, clean.T)
    assert np.linalg.eigvalsh(clean).min() > 0
    # Market mode loads evenly: IPR close to 1/N
    assert pca.inverse_participation_ratio(spec.eigvecs)[0] < 1.5 / N


def test_mst_is_a_spanning_tree():
    rng = np.random.default_rng(2)
    c = _corr(rng.standard_normal((300, 40)))
    g = mst.build_mst(c, [f"s{i}" for i in range(40)])
    assert nx.is_tree(g) and g.number_of_nodes() == 40


def test_distance_metric():
    assert np.allclose(mst.corr_to_distance(np.array([1.0, 0.0, -1.0])), [0, np.sqrt(2), 2])


def test_sector_purity_on_block_structure():
    rng = np.random.default_rng(3)
    T, k, per = 500, 4, 10
    f = rng.standard_normal((T, k))
    x = np.repeat(f, per, axis=1) + rng.standard_normal((T, k * per))
    labels = [f"s{i}" for i in range(k * per)]
    sectors = pd.Series(np.repeat([f"sec{j}" for j in range(k)], per), index=labels)
    g = mst.build_mst(_corr(x), labels)
    res = mst.sector_purity(g, sectors, n_perm=500)
    assert res["observed"] > 0.85 and res["p_value"] < 0.01


def test_varimax_unmixes_rotated_blocks():
    # Three noisy block ("sector") factors of unequal size, mixed by random rotations.
    # (Perfectly identical rows are a degenerate case where the Kaiser update
    # has no gradient to follow; real loadings never look like that.)
    from scipy.stats import special_ortho_group

    rng = np.random.default_rng(0)
    for _ in range(25):
        clean = np.zeros((33, 3))
        start = 0
        for j, size in enumerate([15, 10, 8]):
            clean[start : start + size, j] = rng.uniform(0.5, 0.9, size)
            start += size
        clean += rng.normal(0, 0.05, clean.shape)
        mixed = clean @ special_ortho_group.rvs(3, random_state=rng)
        rec = interpret.varimax(mixed)
        # Every recovered factor matches one of the true blocks
        assert interpret.eigvec_overlap(rec, clean).max(axis=1).min() > 0.99
        # Rotation keeps the subspace and the total variance
        assert np.isclose((rec**2).sum(), (mixed**2).sum())


def test_r2_decomposition_matches_regression():
    rng = np.random.default_rng(4)
    T, N = 2000, 30
    x = rng.standard_normal((T, 1)) + rng.standard_normal((T, N))
    z = (x - x.mean(0)) / x.std(0)
    s = rmt.spectrum(np.corrcoef(z, rowvar=False), T)
    r2 = interpret.r2_decomposition(s.eigvals, s.eigvecs, 1)["market"].values
    f = z @ s.eigvecs[:, 0]
    direct = np.array([np.corrcoef(z[:, i], f)[0, 1] ** 2 for i in range(N)])
    assert np.allclose(r2, direct, atol=1e-6)
