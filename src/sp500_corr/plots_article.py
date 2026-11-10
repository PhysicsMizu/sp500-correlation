"""Figures for the 'what do the PCs mean' article."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .plots import BLUE, DIVERGING, GRID, INK, INK_2, OTHER, SERIES, SURFACE, _save, sector_colors

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "blue", ["#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]
)


def _clean_heatmap(ax):
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def _annotate_cells(ax, m: np.ndarray, fmt: str, threshold: float, dark_above: float):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isfinite(v) and abs(v) >= threshold:
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(v) >= dark_above else INK)


def plot_loadings(tables: dict[str, pd.DataFrame], sectors: pd.Series, titles: dict[str, str], path: Path):
    """Small multiples: top +/- loadings of several PCs, bars colored by sector."""
    colors = sector_colors(sectors)
    keys = list(tables)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, key in zip(axes.flat, keys):
        t = tables[key].iloc[::-1]
        y = np.arange(len(t))
        ax.barh(y, t["loading"], color=[colors[s] for s in t["sector"]], height=0.72,
                edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(y, [f"{tk}  ·  {si[:26]}" for tk, si in zip(t.index, t["sub_industry"])], fontsize=8)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(titles[key], fontsize=11)
        ax.set_xlabel("loading")
    shown = [s for s, c in colors.items() if c != OTHER]
    folded = [s for s, c in colors.items() if c == OTHER]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], label=s) for s in shown]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=OTHER, label="Other: " + ", ".join(folded)))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    _save(fig, path)


def plot_factor_corr(c: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(c.values, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(c.shape[1]), c.columns)
    ax.set_yticks(range(c.shape[0]), c.index)
    _clean_heatmap(ax)
    _annotate_cells(ax, c.values, ".2f", 0.25, 0.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("correlation of daily returns", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("What each eigenportfolio trades like (cells with |ρ| ≥ 0.25 labeled)")
    _save(fig, path)


def plot_activity(act: pd.DataFrame, path: Path):
    m = np.log2(act.values)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(m, cmap=DIVERGING, vmin=-3, vmax=3, aspect="auto")
    ax.set_xticks(range(act.shape[1]), act.columns)
    ax.set_yticks(range(act.shape[0]), act.index)
    _clean_heatmap(ax)
    for i in range(act.shape[0]):
        for j in range(act.shape[1]):
            v = act.values[i, j]
            if v >= 1.8 or v <= 0.35:
                ax.text(j, i, f"{v:.1f}×", ha="center", va="center", fontsize=8,
                        color="#ffffff" if abs(m[i, j]) >= 1.8 else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[-3, -2, -1, 0, 1, 2, 3])
    cb.ax.set_yticklabels(["⅛×", "¼×", "½×", "1×", "2×", "4×", "8×"])
    cb.set_label("variance that year ÷ full-sample average", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("When is each factor loud?")
    _save(fig, path)


def plot_stability(raw: np.ndarray, rot: np.ndarray, rot_labels: list[str], path: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"wspace": 0.75})
    for ax, m, title in [(a, raw, "Raw PCs: first half vs second half"),
                         (b, rot, "Varimax factors: first half vs second half")]:
        im = ax.imshow(m, cmap=SEQ_BLUE, vmin=0, vmax=1)
        _clean_heatmap(ax)
        _annotate_cells(ax, m, ".2f", 0.3, 0.6)
        ax.set_title(title, fontsize=11)
    n = raw.shape[0]
    a.set_xticks(range(n), [f"PC{i + 1}" for i in range(n)], fontsize=8, rotation=90)
    a.set_yticks(range(n), [f"PC{i + 1}" for i in range(n)], fontsize=8)
    a.set_xlabel("2020–2026 eigenvector")
    a.set_ylabel("2015–2020 eigenvector")
    b.set_xticks(range(len(rot_labels)), rot_labels, fontsize=8, rotation=90)
    b.set_yticks(range(len(rot_labels)), rot_labels, fontsize=8)
    b.set_xlabel("2020–2026 factor (matched)")
    cb = fig.colorbar(im, ax=[a, b], shrink=0.7)
    cb.set_label("|cosine similarity|", color=INK_2)
    cb.outline.set_visible(False)
    _save(fig, path)


def plot_r2_by_sector(r2: pd.DataFrame, sectors: pd.Series, path: Path):
    g = r2.groupby(sectors.values).mean().sort_values("signal")
    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(g))
    other = g["signal"] - g["market"]
    ax.barh(y, g["market"], color=BLUE, height=0.65, label="market mode (PC1)")
    ax.barh(y, other, left=g["market"], color=SERIES[1], height=0.65,
            edgecolor=SURFACE, linewidth=2, label="other signal PCs")
    for yi, tot in zip(y, g["signal"]):
        ax.text(tot + 0.01, yi, f"{tot:.0%}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, g.index)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_xlabel("share of a typical stock's daily variance")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("How much of a stock is 'market', 'sector', and 'itself'")
    _save(fig, path)


def plot_cumulative(pc_rets: pd.DataFrame, titles: dict[str, str], path: Path):
    keys = list(titles)
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 2.3 * len(keys)), sharex=True)
    for ax, k in zip(axes, keys):
        cum = pc_rets[k].cumsum() * 100
        ax.plot(cum.index, cum.values, color=BLUE, lw=1.6)
        ax.axhline(0, color=INK_2, lw=0.6)
        ax.set_title(titles[k], fontsize=10)
        ax.set_ylabel("cum. log return, %", fontsize=8)
    fig.tight_layout()
    _save(fig, path)
