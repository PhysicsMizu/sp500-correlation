"""Analysis behind docs/portfolios.md (Part III). Run: uv run sp500-strategies

Train: 2017-01 .. 2022-12 (first 2 years of data are only used as estimation history)
Test:  2023-01 .. last full close. Run once with the pre-registered configs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import backtest as bt
from . import data
from . import plots_strategies as ps
from . import strategies as st

ROOT = Path(__file__).resolve().parents[2]
TRAIN = ("2017-01-01", "2022-12-31")

# Pre-registered configurations: textbook defaults, fixed before any backtest.
PREREG = {
    "Min variance": (lambda: st.min_variance(cap=0.05, clean=True), 504),
    "HRP": (lambda: st.hrp(clean=False, method="single"), 504),
    "Trend filter": (lambda: st.trend_filter(sma_days=210), 504),
    "Lottery (top skew)": (lambda: st.lottery(top=50, window=252), 504),
    "Equal weight": (lambda: st.equal_weight(), 504),
}


def grid():
    """Every variant we looked at in training. All of them count as trials."""
    g = {}
    for cap in (0.03, 0.05, 0.10):
        for clean in (True, False):
            for lb in (252, 504):
                g[f"MV cap={cap:.2f} clean={clean} lb={lb}"] = ("Min variance", lambda c=cap, k=clean: st.min_variance(c, k), lb)
    for clean in (True, False):
        for method in ("single", "ward"):
            for lb in (252, 504):
                g[f"HRP clean={clean} {method} lb={lb}"] = ("HRP", lambda k=clean, m=method: st.hrp(k, m), lb)
    for sma in (105, 147, 210, 252):
        g[f"Trend sma={sma}"] = ("Trend filter", lambda s=sma: st.trend_filter(s), 504)
    for top in (25, 50, 100):
        g[f"Lottery top={top}"] = ("Lottery (top skew)", lambda t=top: st.lottery(t), 504)
    return g


SPY_GRID = {f"SPY trend sma={s} daily": (s, "daily") for s in (50, 100, 150, 200, 250)}
SPY_GRID["SPY trend sma=200 monthly"] = (200, "monthly")


def load(end: str):
    meta = data.get_constituents().set_index("ticker")
    prices = data.get_prices(meta.index.tolist(), "2015-01-01", None)
    universe = data.log_returns(prices).columns
    R = prices[universe].pct_change().iloc[1:]
    bench_px = data.get_prices(["SPY", "RSP", "BIL"], "2014-01-01", None, name="bench")
    bench = bench_px.pct_change()
    fpx = data.get_prices(data.FACTOR_ETFS, "2015-01-01", None, name="factors")
    fr = fpx.pct_change()
    idx = R.index.intersection(bench.dropna().index).intersection(fr.dropna(subset=["SPY"]).index)
    idx = idx[idx <= end]
    R, bench, fr = R.loc[idx], bench.loc[idx], fr.loc[idx]
    factors = pd.DataFrame({
        "Market": fr["SPY"] - bench["BIL"],
        "Size (IWM−SPY)": fr["IWM"] - fr["SPY"],
        "Value (IVE−IVW)": fr["IVE"] - fr["IVW"],
        "Momentum (MTUM−SPY)": fr["MTUM"] - fr["SPY"],
        "Low vol (USMV−SPY)": fr["USMV"] - fr["SPY"],
    })
    # SPY price history including the warm-up before 2015, for the moving average
    spy_px = bench_px["SPY"].dropna()
    spy_px = spy_px[spy_px.index <= end]
    return R, bench, factors, meta.loc[universe], spy_px


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--end", default="2026-11-06", help="last full trading day")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "strategies")
    args = p.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    TEST = ("2023-01-01", args.end)

    R, bench, factors, meta, spy_px = load(args.end)
    rf = bench["BIL"]
    nums: dict = {"universe": R.shape[1], "train": TRAIN, "test": TEST}
    # SPY and BIL daily returns over the full price history (for the trend rule)
    spy_full = spy_px.pct_change()
    bil_px = data.get_prices(["SPY", "RSP", "BIL"], "2014-01-01", None, name="bench")["BIL"]
    bil_full = bil_px.pct_change().reindex(spy_px.index)
    SPY_TREND = "SPY trend (200d)"

    def spy_trend(period, sma=200, check="daily", cost=5.0):
        return st.single_asset_trend(spy_px, spy_full, bil_full, period[0], period[1],
                                     sma_days=sma, check=check, cost_bps=cost)

    def run_period(period, cfgs, cost=5.0):
        res = {}
        for name, (make, lb) in cfgs.items():
            res[name] = bt.run(R, rf, make(), period[0], period[1], lookback=lb, cost_bps=cost)
        return res

    def add_bench(rets: pd.DataFrame, period) -> pd.DataFrame:
        rets = rets.copy()
        rets[SPY_TREND] = spy_trend(period)[0].reindex(rets.index)
        rets["SPY"] = bench["SPY"].reindex(rets.index)
        rets["RSP (real EW S&P)"] = bench["RSP"].reindex(rets.index)
        return rets

    # ---- training: pre-registered configs + the full grid ---------------------------------
    print("training period ...")
    tr = run_period(TRAIN, PREREG)
    tr_rets = add_bench(pd.DataFrame({k: v.returns for k, v in tr.items()}), TRAIN)
    print("grid ...")
    G = grid()
    g_tr, g_te = {}, {}
    for name, (family, make, lb) in G.items():
        g_tr[name] = bt.run(R, rf, make(), *TRAIN, lookback=lb).returns
    for name, (sma, check) in SPY_GRID.items():
        g_tr[name] = spy_trend(TRAIN, sma, check)[0].reindex(tr_rets.index)
    trial_srs = np.array([((x - rf.reindex(x.index)).mean() / (x - rf.reindex(x.index)).std()) for x in g_tr.values()])

    # ---- test: run once ---------------------------------------------------------------------
    print("test period ...")
    te = run_period(TEST, PREREG)
    te_rets = add_bench(pd.DataFrame({k: v.returns for k, v in te.items()}), TEST)
    for name, (family, make, lb) in G.items():
        g_te[name] = bt.run(R, rf, make(), *TEST, lookback=lb).returns
    for name, (sma, check) in SPY_GRID.items():
        g_te[name] = spy_trend(TEST, sma, check)[0].reindex(te_rets.index)

    # ---- summaries ---------------------------------------------------------------------------
    def table(rets, res):
        rows = {}
        for c in rets.columns:
            s = bt.summary(rets[c], rf, market=bench["SPY"])
            ex = (rets[c] - rf.reindex(rets.index)).values
            s["psr"] = bt.probabilistic_sharpe(ex)
            if c in res:
                s["turnover_per_year"] = float(res[c].turnover.sum() / s["years"])
                w = res[c].weights
                s["avg_effective_n"] = float((1 / (w**2).sum(axis=1).replace(0, np.nan)).mean())
                s["avg_invested"] = float(w.sum(axis=1).mean())
            if c == SPY_TREND:
                _, pos = spy_trend((str(rets.index[0].date()), str(rets.index[-1].date())))
                s["switches_per_year"] = float(pos.diff().abs().sum() / s["years"])
                s["avg_invested"] = float(pos.mean())
            rows[c] = s
        return pd.DataFrame(rows).T

    t_train, t_test = table(tr_rets, tr), table(te_rets, te)
    nums["train_table"] = t_train.round(4).to_dict("index")
    nums["test_table"] = t_test.round(4).to_dict("index")

    # deflated Sharpe for each pre-registered config, using the training trials
    dsr = {}
    for c in list(PREREG) + [SPY_TREND]:
        ex = (tr_rets[c] - rf.reindex(tr_rets.index)).values
        dsr[c] = _r(bt.deflated_sharpe(ex, trial_srs), 3)
    nums["deflated_sharpe_train"] = dsr
    nums["n_trials"] = len(trial_srs)

    # ---- inference on the test period ------------------------------------------------------
    print("bootstrap ...")
    boot = bt.bootstrap_metrics(te_rets, rf, "Equal weight", n=args.n_boot)
    boot_spy = bt.bootstrap_metrics(te_rets, rf, "SPY", n=args.n_boot)
    strategies = ["Min variance", "HRP", "Trend filter", "Lottery (top skew)", SPY_TREND]
    inf = {}
    for c in te_rets.columns:
        inf[c] = {
            "sharpe_ci": [_r(boot["sharpe"].loc[c, "lo"], 2), _r(boot["sharpe"].loc[c, "hi"], 2)],
            "vol_ci": [_r(boot["vol"].loc[c, "lo"]), _r(boot["vol"].loc[c, "hi"])],
            "ret_ci": [_r(boot["ret"].loc[c, "lo"]), _r(boot["ret"].loc[c, "hi"])],
            "mdd_ci": [_r(boot["mdd"].loc[c, "lo"]), _r(boot["mdd"].loc[c, "hi"])],
            "d_sharpe_vs_EW_ci": [_r(boot["d_sharpe"].loc[c, "lo"], 2), _r(boot["d_sharpe"].loc[c, "hi"], 2)],
            "d_vol_vs_EW_ci": [_r(boot["d_vol"].loc[c, "lo"]), _r(boot["d_vol"].loc[c, "hi"])],
            "d_ret_vs_EW_ci": [_r(boot["d_ret"].loc[c, "lo"]), _r(boot["d_ret"].loc[c, "hi"])],
            "d_sharpe_vs_SPY_ci": [_r(boot_spy["d_sharpe"].loc[c, "lo"], 2), _r(boot_spy["d_sharpe"].loc[c, "hi"], 2)],
        }
    holm_sh = bt.holm(boot["d_sharpe"].loc[strategies, "p"])
    holm_vol = bt.holm(boot["d_vol"].loc[strategies, "p"])
    for c in strategies:
        inf[c]["p_d_sharpe_vs_EW"] = _r(boot["d_sharpe"].loc[c, "p"], 3)
        inf[c]["p_d_sharpe_vs_EW_holm"] = _r(holm_sh[c], 3)
        inf[c]["p_d_vol_vs_EW"] = _r(boot["d_vol"].loc[c, "p"], 3)
        inf[c]["p_d_vol_vs_EW_holm"] = _r(holm_vol[c], 3)
        # how long until the *active* return vs EW would be significant, at today's IR
        active = (te_rets[c] - te_rets["Equal weight"]).values
        inf[c]["active_IR"] = _r(active.mean() / active.std() * np.sqrt(252), 2)
        inf[c]["min_track_record_years_vs_EW"] = _r(bt.min_track_record_years(active), 1)
        ex = (te_rets[c] - rf.reindex(te_rets.index)).values
        inf[c]["sharpe_se_annual"] = _r(bt.sharpe_se(ex) * np.sqrt(252), 2)
    nums["test_inference"] = inf

    # ---- factor regressions -------------------------------------------------------------------
    regs = {}
    for period, rets in [("train", tr_rets), ("test", te_rets)]:
        regs[period] = {}
        for c in strategies + ["Equal weight"]:
            ex = rets[c] - rf.reindex(rets.index)
            capm = bt.factor_regression(ex, factors[["Market"]])
            full = bt.factor_regression(ex, factors)
            regs[period][c] = {
                "capm_alpha": _r(capm.loc["alpha (ann.)", "coef"]), "capm_alpha_t": _r(capm.loc["alpha (ann.)", "t"], 2),
                "capm_beta": _r(capm.loc["Market", "coef"], 2),
                "ff_alpha": _r(full.loc["alpha (ann.)", "coef"]), "ff_alpha_t": _r(full.loc["alpha (ann.)", "t"], 2),
                "ff_betas": {k: [_r(full.loc[k, "coef"], 2), _r(full.loc[k, "t"], 1)] for k in factors.columns},
                "r2": _r(full.attrs["r2"], 3),
            }
    nums["factor_regressions"] = regs

    # ---- cost sensitivity (test) ------------------------------------------------------------------
    costs = {}
    for bps in (0, 10, 25):
        rr = run_period(TEST, {k: PREREG[k] for k in strategies if k in PREREG}, cost=bps)
        costs[bps] = {k: _r(bt.summary(v.returns, rf)["sharpe"], 3) for k, v in rr.items()}
        costs[bps][SPY_TREND] = _r(bt.summary(spy_trend(TEST, cost=bps)[0].reindex(te_rets.index), rf)["sharpe"], 3)
    costs[5] = {k: _r(t_test.loc[k, "sharpe"], 3) for k in strategies}
    nums["cost_sensitivity_sharpe"] = {str(k): costs[k] for k in sorted(costs)}

    # ---- grid persistence ---------------------------------------------------------------------------
    def sr(x):
        e = x - rf.reindex(x.index)
        return e.mean() / e.std() * np.sqrt(252)

    names = list(G) + list(SPY_GRID)
    fam = {**{k: G[k][0] for k in G}, **{k: SPY_TREND for k in SPY_GRID}}
    gdf = pd.DataFrame({
        "family": [fam[k] for k in names],
        "train_sharpe": [sr(g_tr[k]) for k in names],
        "test_sharpe": [sr(g_te[k]) for k in names],
        "train_vol": [g_tr[k].std() * np.sqrt(252) for k in names],
        "test_vol": [g_te[k].std() * np.sqrt(252) for k in names],
    }, index=names)
    nums["grid"] = gdf.round(3).to_dict("index")
    nums["grid_rank_corr"] = {
        "sharpe": _r(spearmanr(gdf.train_sharpe, gdf.test_sharpe)[0], 2),
        "vol": _r(spearmanr(gdf.train_vol, gdf.test_vol)[0], 2),
    }
    best = gdf.train_sharpe.idxmax()
    nums["grid_best_in_train"] = {"name": best, "train": _r(gdf.loc[best, "train_sharpe"], 2),
                                  "test": _r(gdf.loc[best, "test_sharpe"], 2),
                                  "test_rank": int(gdf.test_sharpe.rank(ascending=False)[best])}

    # ---- survivorship: our EW universe vs the real equal-weight S&P 500 -------------------------------
    surv = {}
    for period, rets in [("train", tr_rets), ("test", te_rets)]:
        a = bt.summary(rets["Equal weight"], rf)["ann_return"]
        b = bt.summary(rets["RSP (real EW S&P)"], rf)["ann_return"]
        surv[period] = {"ew_universe": _r(a), "rsp": _r(b), "gap": _r(a - b)}
    nums["survivorship"] = surv

    # ---- figures --------------------------------------------------------------------------------------
    full = pd.concat([tr_rets, te_rets])
    show = ["Min variance", "HRP", "Trend filter", SPY_TREND, "Lottery (top skew)", "Equal weight", "SPY"]
    ps.plot_wealth(full[show], TEST[0], out / "wealth.png")
    ps.plot_drawdowns(full[show], TEST[0], out / "drawdowns.png")
    ps.plot_risk_return(t_test.loc[show], {c: inf[c] for c in show}, out / "risk_return.png")
    ps.plot_forest(t_test.loc[show], {c: inf[c] for c in show}, out / "sharpe_forest.png")
    ps.plot_grid(gdf, out / "grid.png")
    ps.plot_survivorship(full[["Equal weight", "RSP (real EW S&P)"]], TEST[0], out / "survivorship.png")
    ps.plot_skew(full[show], TRAIN, TEST, out / "skew.png")

    (out / "numbers.json").write_text(json.dumps(nums, indent=2, default=str))
    print(json.dumps({k: nums[k] for k in ["test_table", "test_inference", "deflated_sharpe_train"]}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
