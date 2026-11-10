"""Figures for Part IV (Markov chains)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import BLUE, INK, INK_2, MUTED, ORANGE, SERIES, SURFACE, _save
from .plots_article import SEQ_BLUE, _annotate_cells, _clean_heatmap

COLORS = {
    "SPY": MUTED,
    "HMM regime timing": SERIES[0],
    "HMM vol target": SERIES[2],
    "EWMA vol target": SERIES[1],
    "SPY trend (200d)": SERIES[6],
    "Smoothed HMM timing (look-ahead!)": SERIES[7],
}
STYLE = {"SPY": "--", "Smoothed HMM timing (look-ahead!)": ":"}


def plot_monthly_chain(P: np.ndarray, labels: list[str], nm: pd.DataFrame, path: Path):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    im = a.imshow(P, cmap=SEQ_BLUE, vmin=0, vmax=1)
    a.set_xticks(range(4), labels, rotation=30, ha="right", fontsize=8.5)
    a.set_yticks(range(4), labels, fontsize=8.5)
    _clean_heatmap(a)
    _annotate_cells(a, P, ".2f", 0, 0.5)
    a.set_xlabel("next month")
    a.set_ylabel("this month")
    a.set_title("Transition matrix, 1993–2022", fontsize=11)
    g = nm.groupby("s")
    x = np.arange(4)
    for ax, col, title, unit in [(b, "nvol", "Next month's volatility", "annualized vol, %"),
                                 (c, "nret", "Next month's return", "return, %")]:
        m = g[col].mean().reindex(range(4))
        se = (g[col].std() / np.sqrt(g[col].size())).reindex(range(4))
        ax.bar(x, m, color=[BLUE if i % 2 == 0 else ORANGE for i in x], width=0.6, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(x, m, yerr=1.96 * se, fmt="none", ecolor=INK_2, capsize=4, lw=1.2)
        ax.axhline(0, color=INK_2, lw=0.8)
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{title} given this month's state (±95%)", fontsize=10.5)
    _save(fig, path)


def plot_bic(bic: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ks = sorted(bic)
    vals = [bic[k] for k in ks]
    ax.plot(ks, vals, color=BLUE, marker="o")
    best = ks[int(np.argmin(vals))]
    ax.annotate(f"minimum at K = {best}", (best, min(vals)), xytext=(10, 14), textcoords="offset points",
                fontsize=9, color=INK_2)
    ax.set_xlabel("number of hidden states K")
    ax.set_ylabel("BIC (lower is better)")
    ax.set_title("Choosing the number of regimes (1993–2022)")
    _save(fig, path)


def plot_regimes(spy: pd.Series, p_off: pd.Series, path: Path):
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 5.6), sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    a.plot(spy.index, spy.values, color=INK, lw=1.2)
    a.set_yscale("log")
    a.set_title("SPY (log scale)", fontsize=11)
    risk_off = (p_off > 0.5).reindex(spy.index).fillna(False)
    a.fill_between(spy.index, spy.min(), spy.max(), where=risk_off.values, color=ORANGE, alpha=0.18, lw=0)
    b.fill_between(p_off.index, p_off.values, 0, color=ORANGE, alpha=0.8, lw=0)
    b.axhline(0.5, color=INK_2, lw=0.8, ls="--")
    b.set_ylim(0, 1)
    b.set_ylabel("probability")
    b.set_title("Walk-forward filtered P(next day in a negative-mean regime); shaded above when > 0.5", fontsize=10.5)
    _save(fig, path)


def plot_filtered_vs_smoothed(spy: pd.Series, p_f: pd.Series, p_s: pd.Series, path: Path):
    windows = [("COVID crash", "2020-01-15", "2020-06-30"), ("2025 tariff shock", "2025-02-15", "2025-06-15")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 6), sharex="col", gridspec_kw={"height_ratios": [1, 1.2]})
    for j, (title, a, b) in enumerate(windows):
        s = spy[a:b]
        axes[0, j].plot(s.index, s.values, color=INK, lw=1.4)
        axes[0, j].set_title(f"{title}: SPY", fontsize=10.5)
        ax = axes[1, j]
        ax.plot(p_s[a:b].index, p_s[a:b].values, color=SERIES[7], lw=1.8, label="smoothed (uses the future)")
        ax.plot(p_f[a:b].index, p_f[a:b].values, color=BLUE, lw=1.8, label="filtered (tradable)")
        ax.axhline(0.5, color=INK_2, lw=0.8, ls="--")
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(risk-off regime)")
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1, 0].legend(fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.45))
    _save(fig, path)


def plot_lookahead(R: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    w = (1 + R).cumprod()
    for c in R.columns:
        ax.plot(w.index, w[c], color=COLORS[c], ls=STYLE.get(c, "-"), lw=1.7, label=c)
        ax.annotate(f"{w[c].iloc[-1]:.1f}×", (w.index[-1], w[c].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=INK_2, va="center")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Same model, same rule: the only difference is whether the probabilities peek at the future")
    _save(fig, path)


def plot_wealth_dd(R: pd.DataFrame, path: Path):
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    w = (1 + R).cumprod()
    for c in R.columns:
        a.plot(w.index, w[c], color=COLORS[c], ls=STYLE.get(c, "-"), lw=1.6, label=c)
        dd = w[c] / w[c].cummax() - 1
        b.plot(dd.index, dd.values * 100, color=COLORS[c], ls=STYLE.get(c, "-"), lw=1.1)
    a.set_yscale("log")
    a.legend(fontsize=9, loc="upper left")
    a.set_title("Walk-forward 2003–2026: growth of $1 (log scale)", fontsize=11)
    b.set_ylabel("drawdown, %")
    b.set_title("Drawdowns", fontsize=11)
    _save(fig, path)


def plot_forest(t: pd.DataFrame, inf: dict, bench: str, path: Path, title: str):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    names = list(t.index)
    y = np.arange(len(names))[::-1]
    for yi, c in zip(y, names):
        lo, hi = inf[c]["sharpe_ci"]
        a.plot([lo, hi], [yi, yi], color=COLORS[c], lw=3, solid_capstyle="round")
        a.plot(t.loc[c, "sharpe"], yi, "o", color=COLORS[c], ms=8, mec=SURFACE)
        if c != bench:
            lo, hi = inf[c][f"d_sharpe_vs_{bench}_ci"]
            b.plot([lo, hi], [yi, yi], color=COLORS[c], lw=3, solid_capstyle="round")
            b.plot(t.loc[c, "sharpe"] - t.loc[bench, "sharpe"], yi, "o", color=COLORS[c], ms=8, mec=SURFACE)
    a.set_yticks(y, names)
    for ax in (a, b):
        ax.axvline(0, color=INK_2, lw=0.8)
    a.set_title(f"{title}: Sharpe ratio (95% CI)", fontsize=11)
    b.set_title(f"Sharpe minus {bench} (paired 95% CI)", fontsize=11)
    _save(fig, path)


def plot_vol_forecasts(v_hmm: pd.Series, v_ewma: pd.Series, e_hmm: pd.Series, e_ewma: pd.Series, path: Path):
    fig, (a, b) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    a.plot(v_ewma.index, v_ewma.values * 100, color=COLORS["EWMA vol target"], lw=1.0, label="EWMA (λ = 0.94)")
    a.plot(v_hmm.index, v_hmm.values * 100, color=COLORS["HMM vol target"], lw=1.0, label="HMM (walk-forward)")
    a.set_yscale("log")
    a.set_ylabel("forecast vol, % (log)")
    a.legend(fontsize=9, loc="upper left")
    a.set_title("One-day-ahead volatility forecasts", fontsize=11)
    b.plot(e_ewma.index, e_ewma.values, color=COLORS["EWMA vol target"], lw=1.0, label="EWMA vol target")
    b.plot(e_hmm.index, e_hmm.values, color=COLORS["HMM vol target"], lw=1.0, label="HMM vol target")
    b.set_ylabel("exposure to SPY")
    b.set_ylim(0, 1.05)
    b.legend(fontsize=9, loc="lower left")
    b.set_title("Resulting exposure (target 15%, capped at 100%)", fontsize=11)
    _save(fig, path)
