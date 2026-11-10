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


# --- asymmetry -----------------------------------------------------------------

from sp500_corr import asymmetry as asy  # noqa: E402


def test_sign_flip_swaps_semicovariances_and_tails():
    rng = np.random.default_rng(5)
    y = rng.standard_normal((400, 8)) + rng.standard_normal((400, 1))
    p, n, m = asy.semicovariance(y)
    assert np.allclose(p + n + m, y.T @ y / len(y))  # exact decomposition
    p2, n2, _ = asy.semicovariance(-y)
    assert np.allclose(p, n2) and np.allclose(n, p2)
    lo, hi = asy.tail_dependence(asy.to_uniform(y), 0.1)
    lo2, hi2 = asy.tail_dependence(asy.to_uniform(-y), 0.1)
    assert np.allclose(lo, hi2) and np.allclose(hi, lo2)


def test_ewma_devol_uses_only_the_past():
    rng = np.random.default_rng(6)
    r = pd.DataFrame(rng.standard_normal((300, 3)) * 0.01)
    r2 = r.copy()
    r2.iloc[-1] *= 50  # a shock on the last day
    a = asy.ewma_devol(r)
    b = asy.ewma_devol(r2)
    # Before re-standardization, scaled returns before the shock are untouched;
    # check correlation structure of the earlier rows is identical
    assert np.allclose(np.corrcoef(a.iloc[:-1].values, rowvar=False),
                       np.corrcoef(b.iloc[:-1].values, rowvar=False))


def test_tensor_power_method_identifies_independent_skewed_factors():
    # The ICA identification result: second moments can't tell rotated factors
    # apart, but the coskewness tensor can, if the factors are independent and skewed.
    from scipy.stats import special_ortho_group

    rng = np.random.default_rng(7)
    k, t = 4, 200_000
    s = rng.exponential(size=(t, k)) - 1  # skewness 2 each
    s /= s.std(axis=0)
    mix = special_ortho_group.rvs(k, random_state=rng)
    x = s @ mix.T  # still white: covariance = I, so PCA sees nothing
    kappas, u = asy.tensor_power_decomposition(asy.coskewness(x), k, rng)
    assert interpret.eigvec_overlap(u, mix).max(axis=1).min() > 0.99
    assert np.allclose(kappas, 2, atol=0.1)


# --- backtest engine & strategies ----------------------------------------------------

from sp500_corr import backtest as bt, strategies as st  # noqa: E402


def _toy_returns(T=300, n=4, seed=8):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T)
    R = pd.DataFrame(rng.normal(0.0005, 0.01, (T, n)), index=idx, columns=list("abcd")[:n])
    cash = pd.Series(0.0001, index=idx)
    return R, cash


def test_engine_has_no_lookahead():
    R, cash = _toy_returns()
    seen = []

    def spy(hist):
        seen.append(hist.index[-1])
        return np.full(hist.shape[1], 1 / hist.shape[1])

    res = bt.run(R, cash, spy, "2020-03-01", "2020-12-31", lookback=40)
    rebal = list(res.turnover.index)
    assert len(seen) == len(rebal)
    for last_seen, d in zip(seen, rebal):
        assert last_seen < d  # the strategy never sees the rebalance day's return
    # and the first P&L day is the day *after* the first rebalance
    assert res.returns.index[0] > rebal[0]


def test_engine_accounting():
    R, cash = _toy_returns()
    res = bt.run(R, cash, lambda h: np.array([1.0, 0, 0, 0]), "2020-03-01", "2020-06-30",
                 lookback=20, cost_bps=10)
    first = res.returns.index[0]
    # 100% in asset a: gross = r_a; first day pays cost on 2.0 of two-way turnover
    assert np.isclose(res.gross[first], R.loc[first, "a"])
    assert np.isclose(res.returns[first], R.loc[first, "a"] - 2.0 * 10 / 1e4)
    # all cash -> earns the cash rate exactly, no turnover after the first trade
    res0 = bt.run(R, cash, lambda h: np.zeros(4), "2020-03-01", "2020-06-30", lookback=20)
    assert np.allclose(res0.returns, 0.0001) and np.allclose(res0.turnover, 0)


def test_min_variance_respects_constraints_and_theory():
    rng = np.random.default_rng(9)
    vols = np.array([0.01, 0.02, 0.03, 0.04])
    hist = pd.DataFrame(rng.standard_normal((5000, 4)) * vols)
    w = st.min_variance(cap=1.0, clean=False)(hist)
    # uncorrelated => inverse variance, with the strategy's own (EWMA) variances
    v = st._ewma_vol(hist) ** 2
    expected = (1 / v) / (1 / v).sum()
    assert np.allclose(w, expected, atol=0.02)
    w_cap = st.min_variance(cap=0.4, clean=False)(hist)
    assert w_cap.max() <= 0.4 + 1e-6 and np.isclose(w_cap.sum(), 1)


def test_hrp_balances_risk_across_clusters():
    rng = np.random.default_rng(10)
    T = 3000
    # cluster A: 6 correlated stocks; cluster B: 2 correlated stocks, same vol
    fa, fb = rng.standard_normal((T, 1)), rng.standard_normal((T, 1))
    x = np.hstack([fa + 0.3 * rng.standard_normal((T, 6)), fb + 0.3 * rng.standard_normal((T, 2))]) * 0.01
    w = st.hrp()(pd.DataFrame(x))
    assert np.isclose(w.sum(), 1) and (w > 0).all()
    # roughly equal capital per cluster despite 6 vs 2 names (each cluster is ~one bet)
    assert 0.35 < w[:6].sum() < 0.65


def test_sharpe_se_matches_normal_formula():
    rng = np.random.default_rng(11)
    ex = rng.normal(0.05, 1, 20000)
    sr = ex.mean() / ex.std()
    assert np.isclose(bt.sharpe_se(ex), np.sqrt((1 + sr**2 / 2) / len(ex)), rtol=0.05)


def test_holm_adjustment():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})
    adj = bt.holm(p)
    assert np.isclose(adj["a"], 0.03) and np.isclose(adj["c"], 0.06) and np.isclose(adj["b"], 0.06)


def test_single_asset_trend_timing_and_costs():
    idx = pd.bdate_range("2020-01-01", periods=60)
    # price rises for 30 days, then crashes and falls for 30 days
    price = pd.Series(np.r_[np.linspace(100, 130, 30), np.linspace(100, 70, 30)], index=idx)
    risky = price.pct_change().fillna(0)
    cash = pd.Series(0.0, index=idx)
    r, pos = st.single_asset_trend(price, risky, cash, str(idx[0].date()), str(idx[-1].date()),
                                   sma_days=10, cost_bps=0)
    above = price > price.rolling(10).mean()
    first_below = above[above.index > idx[29]].idxmin()  # first day the close is below the MA
    k = idx.get_loc(first_below)
    # still long on the crash day itself and the day after; out two days after the signal
    assert pos.iloc[k] == 1 and pos.iloc[k + 1] == 1 and pos.iloc[k + 2] == 0
    # with costs, a switch costs 2 x bps on the day the new position starts
    r5, _ = st.single_asset_trend(price, risky, cash, str(idx[0].date()), str(idx[-1].date()),
                                  sma_days=10, cost_bps=5)
    assert np.isclose(r.iloc[k + 2] - r5.iloc[k + 2], 2 * 5 / 1e4)
