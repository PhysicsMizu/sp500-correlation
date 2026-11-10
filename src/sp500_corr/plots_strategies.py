"""Figures for Part III (portfolios)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import GRID, INK, INK_2, MUTED, SERIES, SURFACE, _save

COLORS = {
    "Min variance": SERIES[0],
    "Trend filter": SERIES[1],
    "HRP": SERIES[2],
    "Lottery (top skew)": SERIES[4],
    "Equal weight": "#52514e",
    "SPY": MUTED,
    "RSP (real EW S&P)": SERIES[3],
    "SPY trend (200d)": SERIES[6],
}
STYLE = {"SPY": "--"}
# label placement for the risk-return scatter: (dx, dy, horizontal alignment)
LABEL_OFFSETS = {"HRP": (-8, -16, "right"), "SPY trend (200d)": (-6, 8, "right"),
                 "Min variance": (8, -16, "left"), "Trend filter": (8, -16, "left"),
                 "Equal weight": (8, 6, "left")}
FAMILY_COLORS = {"Min variance": SERIES[0], "Trend filter": SERIES[1], "HRP": SERIES[2],
                 "Lottery (top skew)": SERIES[4], "SPY trend (200d)": SERIES[6]}


def _shade_test(ax, test_start):
    ax.axvspan(pd.Timestamp(test_start), ax.get_xlim()[1], color=GRID, alpha=0.35, lw=0, zorder=0)


def plot_wealth(rets: pd.DataFrame, test_start: str, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, sub, title in [(axes[0], rets, "Full period, growth of $1 (log scale)"),
                           (axes[1], rets.loc[test_start:], "Test period only (2023 →), rebased")]:
        w = (1 + sub).cumprod()
        for c in sub.columns:
            ax.plot(w.index, w[c], color=COLORS[c], ls=STYLE.get(c, "-"), lw=1.6, label=c)
            ax.annotate(f"{w[c].iloc[-1]:.2f}", (w.index[-1], w[c].iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=8, color=INK_2, va="center")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].axvline(pd.Timestamp(test_start), color=INK_2, lw=0.8, ls=":")
    axes[0].annotate("train | test", (pd.Timestamp(test_start), axes[0].get_ylim()[1]), xytext=(-30, -14),
                     textcoords="offset points", fontsize=9, color=INK_2)
    axes[0].legend(fontsize=8.5, loc="upper left")
    _save(fig, path)


def plot_drawdowns(rets: pd.DataFrame, test_start: str, path: Path):
    fig, axes = plt.subplots(len(rets.columns), 1, figsize=(11, 1.6 * len(rets.columns)), sharex=True, sharey=True)
    for ax, c in zip(axes, rets.columns):
        w = (1 + rets[c]).cumprod()
        dd = w / w.cummax() - 1
        ax.fill_between(dd.index, dd.values * 100, 0, color=COLORS[c], alpha=0.8, lw=0)
        ax.axvline(pd.Timestamp(test_start), color=INK_2, lw=0.8, ls=":")
        ax.set_title(f"{c}  (worst {dd.min():.0%})", fontsize=9.5)
        ax.set_ylabel("%", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_risk_return(t: pd.DataFrame, inf: dict, path: Path):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for c in t.index:
        v, r = t.loc[c, "ann_vol"] * 100, t.loc[c, "ann_return"] * 100
        vl, vh = np.array(inf[c]["vol_ci"]) * 100
        rl, rh = np.array(inf[c]["ret_ci"]) * 100
        ax.errorbar(v, r, xerr=[[v - vl], [vh - v]], yerr=[[r - rl], [rh - r]], fmt="o", ms=8,
                    color=COLORS[c], ecolor=COLORS[c], elinewidth=1.5, capsize=3, label=c)
        dx, dy, ha = LABEL_OFFSETS.get(c, (8, 6, "left"))
        ax.annotate(c, (v, r), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK, ha=ha)
    ax.set_xlabel("annualized volatility, % (95% CI)")
    ax.set_ylabel("annualized return, % (95% CI)")
    ax.set_title("Test period: risk is measurable, return barely is")
    _save(fig, path)


def plot_forest(t: pd.DataFrame, inf: dict, path: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    names = list(t.index)
    y = np.arange(len(names))[::-1]
    for yi, c in zip(y, names):
        lo, hi = inf[c]["sharpe_ci"]
        a.plot([lo, hi], [yi, yi], color=COLORS[c], lw=3, solid_capstyle="round")
        a.plot(t.loc[c, "sharpe"], yi, "o", color=COLORS[c], ms=8, mec=SURFACE)
        if c != "Equal weight":
            lo, hi = inf[c]["d_sharpe_vs_EW_ci"]
            d = t.loc[c, "sharpe"] - t.loc["Equal weight", "sharpe"]
            b.plot([lo, hi], [yi, yi], color=COLORS[c], lw=3, solid_capstyle="round")
            b.plot(d, yi, "o", color=COLORS[c], ms=8, mec=SURFACE)
    a.set_yticks(y, names)
    a.axvline(0, color=INK_2, lw=0.8)
    b.axvline(0, color=INK_2, lw=0.8)
    a.set_title("Sharpe ratio, test period (95% CI)", fontsize=11)
    b.set_title("Sharpe minus equal weight (paired 95% CI)", fontsize=11)
    _save(fig, path)


def plot_grid(g: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [(axes[0], "sharpe", "Sharpe ratio"), (axes[1], "vol", "Volatility")]:
        for fam, sub in g.groupby("family"):
            ax.scatter(sub[f"train_{col}"], sub[f"test_{col}"], s=40, color=FAMILY_COLORS[fam],
                       label=fam, edgecolors=SURFACE, linewidths=0.8)
        lo = min(g[f"train_{col}"].min(), g[f"test_{col}"].min())
        hi = max(g[f"train_{col}"].max(), g[f"test_{col}"].max())
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--")
        ax.set_xlabel(f"train (2017–2022) {title.lower()}")
        ax.set_ylabel(f"test (2023–2026) {title.lower()}")
        ax.set_title(f"{title}: does the training ranking survive?", fontsize=11)
    axes[0].legend(fontsize=8.5)
    _save(fig, path)


def plot_survivorship(rets: pd.DataFrame, test_start: str, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = (1 + rets).cumprod()
    for c in rets.columns:
        ax.plot(w.index, w[c], color=COLORS[c], lw=1.8, label=c)
    ax.axvline(pd.Timestamp(test_start), color=INK_2, lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Survivorship bias: our 'equal weight' vs the real equal-weight S&P 500 ETF")
    _save(fig, path)


def plot_skew(rets: pd.DataFrame, train, test, path: Path):
    def mskew(r):
        m = (1 + r).groupby([r.index.year, r.index.month]).prod() - 1
        z = (m - m.mean()) / m.std()
        return float((z**3).mean())

    rows = {c: [mskew(rets[c].loc[train[0]:train[1]]), mskew(rets[c].loc[test[0]:test[1]])] for c in rets.columns}
    df = pd.DataFrame(rows, index=["train", "test"]).T
    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = np.arange(len(df))
    ax.bar(x - 0.19, df["train"], width=0.36, color=SERIES[0], label="train 2017–2022", edgecolor=SURFACE, linewidth=2)
    ax.bar(x + 0.19, df["test"], width=0.36, color=SERIES[1], label="test 2023–2026", edgecolor=SURFACE, linewidth=2)
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, df.index, fontsize=9)
    ax.set_ylabel("skewness of monthly returns")
    ax.legend(fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.set_title("Monthly return skewness")
    _save(fig, path)
