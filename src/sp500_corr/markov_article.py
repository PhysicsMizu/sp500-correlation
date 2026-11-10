"""Analysis behind docs/markov.md (Part IV). Run: uv run sp500-markov

Periods
  HMM / chain estimation (training):   1993-01 .. 2022-12
  Long walk-forward test (timing):     2003-01 .. 2026-11 (HMM refit each January on prior data only)
  Single test, as in Part III:         2023-01 .. 2026-11
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal

from . import backtest as bt
from . import data
from . import markov as mk
from . import plots_markov as pm
from . import strategies as st

ROOT = Path(__file__).resolve().parents[2]
TRAIN_END = "2022-12-31"
TEST = ("2023-01-01", "2026-11-06")
WF_START = 2003
SIGMA_TARGET = 0.15      # pre-registered vol target (annual), same for HMM and EWMA
COST_BPS = 5.0


def _r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def exposure_returns(expo: pd.Series, risky: pd.Series, cash: pd.Series, start, end,
                     cost_bps: float = COST_BPS) -> tuple[pd.Series, pd.Series]:
    """expo[t] uses information up to the close of t. Same timing as Part III:
    traded at the close of t+1, earns from t+2. Costs on |change in exposure| x 2."""
    pos = expo.shift(2)
    days = risky.index[(risky.index >= start) & (risky.index <= end)]
    pos = pos.reindex(days).ffill().fillna(0.0).clip(0, 1)
    trades = pos.diff().abs().fillna(pos.iloc[0])
    r = pos * risky.reindex(days) + (1 - pos) * cash.reindex(days) - trades * 2 * cost_bps / 1e4
    return r, pos


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "markov")
    args = p.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    nums: dict = {}

    px = data.get_prices(["SPY", "^IRX"], "1993-01-01", None, name="spylong")
    spy = px["SPY"].dropna()
    spy = spy[spy.index <= TEST[1]]
    logr = np.log(spy).diff().dropna()
    simple = spy.pct_change().dropna()
    irx = px["^IRX"].reindex(spy.index).ffill()
    cash = ((1 + irx / 100) ** (1 / 252) - 1).reindex(simple.index).fillna(0)
    train = logr[:TRAIN_END]

    # ---- 1. observable chains -------------------------------------------------------------
    print("1/6 observable chains")
    q = np.quantile(train, [1 / 3, 2 / 3])
    s_daily = np.digitize(train.values, q)
    nxt = pd.DataFrame({"s": s_daily[:-1], "r": train.values[1:]})
    nums["daily_chain"] = {
        "P": np.round(mk.transition_matrix(s_daily, 3), 3).tolist(),
        "order1": mk.markov_order_test(s_daily, 3, 1),
        "order2": mk.markov_order_test(s_daily, 3, 2),
        "next_abs_ret_pct": nxt.groupby("s").r.apply(lambda x: x.abs().mean() * 100).round(3).tolist(),
        "next_mean_ann_pct": nxt.groupby("s").r.apply(lambda x: x.mean() * 252 * 100).round(1).tolist(),
        "kruskal_next_ret_p": _r(kruskal(*[g.r for _, g in nxt.groupby("s")]).pvalue),
        "kruskal_next_abs_p": float(kruskal(*[g.r.abs() for _, g in nxt.groupby("s")]).pvalue),
    }

    month = logr.groupby([logr.index.year, logr.index.month]).agg(ret="sum", vol="std")
    month.index = pd.PeriodIndex([f"{y}-{m:02d}" for y, m in month.index], freq="M")
    month["vol"] *= np.sqrt(252)
    mt = month.loc[:"2022-12"]
    vmed = mt.vol.median()

    def mstate(df):
        return ((df.ret.values > 0).astype(int) * 2 + (df.vol.values > vmed).astype(int))

    labels = ["down · calm", "down · volatile", "up · calm", "up · volatile"]
    sm = mstate(mt)
    P = mk.transition_matrix(sm, 4)
    nm = pd.DataFrame({"s": sm[:-1], "nret": mt.ret.values[1:] * 100, "nvol": mt.vol.values[1:] * 100})
    rng = np.random.default_rng(0)

    def boot_ci(x, n=2000):
        b = [rng.choice(x, len(x)).mean() for _ in range(n)]
        return np.percentile(b, [2.5, 97.5]).round(2).tolist()

    by_state = {}
    for s, g in nm.groupby("s"):
        by_state[labels[s]] = {"n": len(g), "next_ret_pct": _r(g.nret.mean(), 2), "next_ret_ci": boot_ci(g.nret.values),
                               "next_vol_pct": _r(g.nvol.mean(), 1), "next_vol_ci": boot_ci(g.nvol.values)}
    # out of sample: same thresholds, test months
    mte = month.loc["2023-01":]
    ste = mstate(mte)
    nte = pd.DataFrame({"s": ste[:-1], "nvol": mte.vol.values[1:] * 100, "nret": mte.ret.values[1:] * 100})
    vol_only = (mt.vol.values > vmed).astype(int)
    sign_only = (mt.ret.values > 0).astype(int)
    nums["monthly_chain"] = {
        "labels": labels, "vol_median_ann_pct": _r(vmed * 100, 1),
        "P": np.round(P, 3).tolist(), "stationary": np.round(mk.stationary_distribution(P), 3).tolist(),
        "durations_months": np.round(mk.expected_durations(P), 2).tolist(),
        "order1": mk.markov_order_test(sm, 4, 1), "order2": mk.markov_order_test(sm, 4, 2),
        "vol_only_order1": mk.markov_order_test(vol_only, 2, 1),
        "sign_only_order1": mk.markov_order_test(sign_only, 2, 1),
        "by_state": by_state,
        "kruskal_next_ret_p": _r(kruskal(*[g.nret for _, g in nm.groupby("s")]).pvalue),
        "kruskal_next_vol_p": float(kruskal(*[g.nvol for _, g in nm.groupby("s")]).pvalue),
        "test_next_vol_by_calm_vs_volatile": {
            "after_calm": _r(nte[nte.s % 2 == 0].nvol.mean(), 1), "after_volatile": _r(nte[nte.s % 2 == 1].nvol.mean(), 1),
            "n_calm": int((nte.s % 2 == 0).sum()), "n_volatile": int((nte.s % 2 == 1).sum())},
    }
    pm.plot_monthly_chain(P, labels, nm, out / "monthly_chain.png")

    # the one *return* effect: next-day reversal after a bottom-tercile day. Real? Tradable?
    def rev_gap(x):
        s_ = np.digitize(x[:-1], q)
        nx = x[1:]
        return (nx[s_ == 0].mean() - nx[s_ == 1].mean()) * 252

    xs = train.values
    bs = [rev_gap(xs[bt.stationary_bootstrap_indices(len(xs), 21, rng)]) for _ in range(2000)]
    sig = (logr < q[0]).astype(float)
    rev = {"gap_train_ann_pct": _r(rev_gap(xs) * 100, 1), "z_block_bootstrap": _r(rev_gap(xs) / np.std(bs), 2),
           "gap_test_ann_pct": _r(rev_gap(logr[TEST[0]:].values) * 100, 1)}
    for lag, label in [(1, "trade at the down day's close"), (2, "one day later (Part III timing)")]:
        pos = sig.shift(lag).reindex(simple.index).fillna(0)
        for per_name, (a, b) in {"2003-2022": (f"{WF_START}-01-01", TRAIN_END), "2023-2026": TEST}.items():
            d = simple.loc[a:b]
            pp = pos.loc[d.index]
            trd = pp.diff().abs().fillna(0)
            g = pp * d + (1 - pp) * cash.loc[d.index]
            n5 = g - trd * 2 * 5 / 1e4
            excess = (g - cash.loc[d.index]).sum()
            rev[f"{label} | {per_name}"] = {
                "gross_sharpe": _r(bt.summary(g, cash)["sharpe"], 2), "net_sharpe_5bps": _r(bt.summary(n5, cash)["sharpe"], 2),
                "gross_ann_return": _r(bt.summary(g, cash)["ann_return"], 3), "net_ann_return_5bps": _r(bt.summary(n5, cash)["ann_return"], 3),
                "time_in_market": _r(pp.mean(), 2), "switches_per_year": _r(trd.sum() / (len(d) / 252), 0),
                "breakeven_bps_per_side": _r(excess / (2 * trd.sum()) * 1e4, 2)}
    nums["daily_reversal"] = rev

    # ---- 2. HMM: how many states? -----------------------------------------------------------
    print("2/6 HMM model selection")
    fits = {k: mk.fit_hmm(train.values, k, n_init=4) for k in range(1, 7)}
    bic = {k: f.bic(len(train)) for k, f in fits.items()}
    nums["hmm_bic"] = {k: _r(v, 0) for k, v in bic.items()}
    nums["hmm_states"] = {k: {"ann_mu_pct": np.round(f.mu * 252 * 100, 1).tolist(),
                              "ann_vol_pct": np.round(f.sigma * np.sqrt(252) * 100, 1).tolist(),
                              "duration_days": np.round(1 / (1 - np.diag(f.A)), 0).tolist(),
                              "stationary": np.round(mk.stationary_distribution(f.A), 3).tolist()}
                          for k, f in fits.items() if k in (2, 3, 4)}
    pm.plot_bic(bic, out / "bic.png")

    # ---- 3. filtered vs smoothed: the look-ahead trap ---------------------------------------------
    print("3/6 filtered vs smoothed")
    full = mk.fit_hmm(logr.values, 4, n_init=4)  # fitted on everything, deliberately
    filt, _ = full.filter(logr.values)
    smth = full.smooth(logr.values)
    off = full.mu < 0
    p_f = pd.Series(filt[:, off].sum(axis=1), index=logr.index)
    p_s = pd.Series(smth[:, off].sum(axis=1), index=logr.index)
    pm.plot_filtered_vs_smoothed(spy, p_f, p_s, out / "filtered_vs_smoothed.png")
    # isolate the smoothing effect: identical parameters, filtered vs smoothed
    iso = {}
    for name, pr in [("filtered", p_f), ("smoothed", p_s)]:
        x, _ = exposure_returns((pr < 0.5).astype(float), simple, cash, f"{WF_START}-01-01", TEST[1])
        sm_ = bt.summary(x, cash)
        iso[name] = {"sharpe": _r(sm_["sharpe"], 3), "ann_return": _r(sm_["ann_return"]), "max_drawdown": _r(sm_["max_drawdown"], 3),
                     "switches": int((pr[f"{WF_START}":] > 0.5).astype(int).diff().abs().sum())}
    nums["smoothing_isolated_same_params"] = iso
    # how early does the smoothed probability "know"?
    lead = {}
    for name, (a, b) in {"COVID 2020": ("2020-02-01", "2020-04-30"), "Tariffs 2025": ("2025-03-01", "2025-05-31")}.items():
        f_cross = p_f[a:b][p_f[a:b] > 0.5].index.min()
        s_cross = p_s[a:b][p_s[a:b] > 0.5].index.min()
        lead[name] = {"filtered_first_above_50": str(f_cross.date()) if pd.notna(f_cross) else None,
                      "smoothed_first_above_50": str(s_cross.date()) if pd.notna(s_cross) else None,
                      "trading_days_earlier": int(len(logr[s_cross:f_cross]) - 1) if pd.notna(f_cross) and pd.notna(s_cross) else None}
    nums["lookahead_lead"] = lead

    # ---- 4. walk-forward strategies 2003-2026 --------------------------------------------------------
    print("4/6 walk-forward (refits every year)")
    wf = mk.walk_forward_hmm(logr, WF_START)
    nums["wf_k_by_year"] = wf["k"]
    ewma_var = pd.Series(mk.ewma_variance_forecast(logr.values), index=logr.index)
    start, end = f"{WF_START}-01-01", TEST[1]
    tgt = SIGMA_TARGET / np.sqrt(252)
    expos = {
        "HMM regime timing": (wf["p_off"] < 0.5).astype(float),
        "HMM vol target": (tgt / np.sqrt(wf["var"])).clip(upper=1.0),
        "EWMA vol target": (tgt / np.sqrt(ewma_var)).clip(upper=1.0),
        "Smoothed HMM timing (look-ahead!)": (p_s < 0.5).astype(float),
    }
    rets, poss = {}, {}
    for name, e in expos.items():
        rets[name], poss[name] = exposure_returns(e, simple, cash, start, end)
    tr_ret, tr_pos = st.single_asset_trend(spy, simple, cash, start, end, sma_days=200, cost_bps=COST_BPS)
    rets["SPY trend (200d)"], poss["SPY trend (200d)"] = tr_ret, tr_pos
    rets["SPY"] = simple.loc[start:end]
    R = pd.DataFrame(rets).dropna()
    honest = ["HMM regime timing", "HMM vol target", "EWMA vol target", "SPY trend (200d)", "SPY"]
    cash_r = cash.reindex(R.index)

    def tables(Rp, cols):
        rows = {}
        for c in cols:
            s = bt.summary(Rp[c], cash_r, market=Rp["SPY"])
            if c in poss:
                pp = poss[c].reindex(Rp.index)
                s["avg_exposure"] = float(pp.mean())
                s["turnover_per_year"] = float(pp.diff().abs().sum() / s["years"])
            rows[c] = s
        return pd.DataFrame(rows).T

    def inference(Rp, cols, bench):
        b = bt.bootstrap_metrics(Rp[cols], cash_r.reindex(Rp.index), bench, n=args.n_boot)
        strategies = [c for c in cols if c != bench]
        hs, hv = bt.holm(b["d_sharpe"].loc[strategies, "p"]), bt.holm(b["d_vol"].loc[strategies, "p"])
        res = {}
        for c in cols:
            res[c] = {"sharpe_ci": [_r(b["sharpe"].loc[c, "lo"], 2), _r(b["sharpe"].loc[c, "hi"], 2)],
                      "vol_ci": [_r(b["vol"].loc[c, "lo"]), _r(b["vol"].loc[c, "hi"])],
                      "mdd_ci": [_r(b["mdd"].loc[c, "lo"]), _r(b["mdd"].loc[c, "hi"])],
                      f"d_sharpe_vs_{bench}_ci": [_r(b["d_sharpe"].loc[c, "lo"], 2), _r(b["d_sharpe"].loc[c, "hi"], 2)],
                      f"d_ret_vs_{bench}_ci": [_r(b["d_ret"].loc[c, "lo"]), _r(b["d_ret"].loc[c, "hi"])]}
            if c in strategies:
                res[c]["p_d_sharpe"] = _r(b["d_sharpe"].loc[c, "p"], 3)
                res[c]["p_d_sharpe_holm"] = _r(hs[c], 3)
                res[c]["p_d_vol_holm"] = _r(hv[c], 3)
        return res

    t_wf = tables(R, list(R.columns))
    nums["wf_table"] = t_wf.round(4).to_dict("index")
    nums["wf_inference_vs_SPY"] = inference(R, honest, "SPY")
    # the head-to-head that matters: does the Markov structure beat a plain EWMA?
    h2h = bt.bootstrap_metrics(R[["HMM vol target", "EWMA vol target"]], cash_r, "EWMA vol target", n=args.n_boot)
    nums["wf_hmm_vs_ewma_vt"] = {"d_sharpe_ci": [_r(h2h["d_sharpe"].loc["HMM vol target", "lo"], 3), _r(h2h["d_sharpe"].loc["HMM vol target", "hi"], 3)],
                                 "p": _r(h2h["d_sharpe"].loc["HMM vol target", "p"], 3),
                                 "corr_daily": _r(R["HMM vol target"].corr(R["EWMA vol target"]), 3),
                                 "corr_exposure": _r(poss["HMM vol target"].corr(poss["EWMA vol target"]), 3)}
    # minimum track record for the timing rule's active return vs SPY
    for c in ["HMM regime timing", "HMM vol target", "EWMA vol target", "SPY trend (200d)"]:
        a = (R[c] - R["SPY"]).values
        nums["wf_table"][c]["active_IR_vs_SPY"] = _r(a.mean() / a.std() * np.sqrt(252), 2)
        nums["wf_table"][c]["min_track_record_years_vs_SPY"] = _r(bt.min_track_record_years(a), 1)

    # volatility forecast accuracy (out of sample, 2003-2026 and test)
    vf = {}
    for label, (a, b) in {"2003-2026": (start, end), "test 2023-2026": TEST}.items():
        idx = wf["var"].loc[a:b].index[:-1]
        realized = logr.shift(-1).reindex(idx) ** 2
        lh = mk.qlike(realized.values, wf["var"].reindex(idx).values)
        le = mk.qlike(realized.values, ewma_var.reindex(idx).values)
        vf[label] = {"qlike_hmm": _r(lh.mean()), "qlike_ewma": _r(le.mean()), **mk.diebold_mariano(lh, le)}
    nums["vol_forecast"] = vf
    nums["vol_forecast_corr"] = _r(np.corrcoef(np.sqrt(wf["var"]), np.sqrt(ewma_var.reindex(wf["var"].index)))[0, 1], 3)

    pm.plot_regimes(spy.loc[start:end], wf["p_off"], out / "regimes.png")
    pm.plot_lookahead(R[["SPY", "HMM regime timing", "Smoothed HMM timing (look-ahead!)"]], out / "lookahead.png")
    pm.plot_wealth_dd(R[honest], out / "wf_wealth.png")
    pm.plot_forest(t_wf.loc[honest], nums["wf_inference_vs_SPY"], "SPY", out / "wf_forest.png",
                   "Walk-forward 2003–2026")
    pm.plot_vol_forecasts(np.sqrt(wf["var"] * 252), np.sqrt(ewma_var.reindex(wf["var"].index) * 252),
                          poss["HMM vol target"], poss["EWMA vol target"], out / "vol_forecasts.png")

    # ---- 5. the Part III test window ------------------------------------------------------------
    print("5/6 test window 2023-2026")
    Rt = R.loc[TEST[0]:TEST[1]]
    nums["test_table"] = tables(Rt, list(Rt.columns)).round(4).to_dict("index")
    nums["test_inference_vs_SPY"] = inference(Rt, honest, "SPY")
    for name, (a, b) in {"2003-2022 (pre-test)": (start, TRAIN_END), "2008 crisis (2007-10 to 2009-03)": ("2007-10-01", "2009-03-31"),
                         "COVID (2020-02-19 to 2020-12-31)": ("2020-02-19", "2020-12-31"), "2022 bear": ("2022-01-01", "2022-12-31")}.items():
        nums.setdefault("episodes", {})[name] = {c: _r((1 + R[c].loc[a:b]).prod() - 1, 3) for c in honest + ["Smoothed HMM timing (look-ahead!)"]}

    # ---- 6. regime-aware minimum variance (cross-section, Part III setup) --------------------------
    print("6/6 regime-aware minimum variance")
    from .strat_article import load
    Ru, bench, factors, meta, _ = load(TEST[1])
    rf = bench["BIL"]
    hmm2 = mk.fit_hmm(train.values, 2, n_init=4)          # K=2: each regime needs enough days
    f2, _ = hmm2.filter(logr.values)
    probs = pd.DataFrame(f2, index=logr.index)
    BASE = "Min variance, window vols (no regimes)"
    mv_cfg = {"Min variance (Part III, EWMA vols)": st.min_variance(cap=0.05, clean=True),
              BASE: st.min_variance_window(cap=0.05),
              "Regime-mixture min variance": st.regime_min_variance(probs, hmm2.A, cap=0.05)}
    mvres = {}
    for period_name, (a, b) in {"train": ("2017-01-01", TRAIN_END), "test": TEST}.items():
        rr = {k: bt.run(Ru, rf, f, a, b, lookback=504, cost_bps=COST_BPS) for k, f in mv_cfg.items()}
        dfr = pd.DataFrame({k: v.returns for k, v in rr.items()})
        # fair comparison: same vol estimator, with vs without regimes
        bb = bt.bootstrap_metrics(dfr, rf, BASE, n=args.n_boot)
        mvres[period_name] = {k: {**{m: _r(v, 4) for m, v in bt.summary(dfr[k], rf).items()},
                                  "turnover_per_year": _r(rr[k].turnover.sum() / (len(dfr) / 252), 2),
                                  "d_sharpe_ci": [_r(bb["d_sharpe"].loc[k, "lo"], 2), _r(bb["d_sharpe"].loc[k, "hi"], 2)],
                                  "d_vol_ci": [_r(bb["d_vol"].loc[k, "lo"]), _r(bb["d_vol"].loc[k, "hi"])],
                                  "corr_with_base": _r(dfr[k].corr(dfr[BASE]), 3)}
                              for k in dfr.columns}
    nums["regime_mv"] = mvres
    nums["hmm2_states"] = {"ann_mu_pct": np.round(hmm2.mu * 252 * 100, 1).tolist(),
                           "ann_vol_pct": np.round(hmm2.sigma * np.sqrt(252) * 100, 1).tolist()}

    (out / "numbers.json").write_text(json.dumps(nums, indent=2, default=str))
    print(json.dumps({k: nums[k] for k in ["wf_table", "wf_hmm_vs_ewma_vt", "vol_forecast"]}, indent=1, default=str)[:5000])


if __name__ == "__main__":
    main()
