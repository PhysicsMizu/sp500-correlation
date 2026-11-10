"""Figures for Part II (asymmetry)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plots import BLUE, DIVERGING, INK, INK_2, MUTED, ORANGE, OTHER, SERIES, SURFACE, _save, sector_colors
from .plots_article import _annotate_cells, _clean_heatmap

DOWN, UP, NULL = ORANGE, BLUE, "#c3c2b7"


def plot_exceedance(curves: dict[str, dict], path: Path):
    """curves[label] = {'data': df(down, up), 'gauss': df, 'band': df(lo, hi)}"""
    fig, axes = plt.subplots(1, len(curves), figsize=(12, 4.4), sharey=True)
    for ax, (label, c) in zip(axes, curves.items()):
        x = c["data"].index * 100
        ax.fill_between(x, c["band"]["lo"], c["band"]["hi"], color=NULL, alpha=0.45, lw=0,
                        label="sign-flip null (95%)")
        g = c["gauss"][["down", "up"]].mean(axis=1)
        ax.plot(x, g, color=MUTED, lw=1.4, ls="--", label="Gaussian twin")
        ax.plot(x, c["data"]["down"], color=DOWN, marker="o", ms=4, label="down-market days")
        ax.plot(x, c["data"]["up"], color=UP, marker="o", ms=4, label="up-market days")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("market in its bottom / top q% of days")
    axes[0].set_ylabel("mean pairwise correlation")
    axes[0].legend(fontsize=8.5, loc="upper left")
    _save(fig, path)


def plot_semicov(eig: np.ndarray, null: np.ndarray, a_vec: np.ndarray, pc1: np.ndarray, path: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1]})
    k = np.arange(1, len(eig) + 1)
    lo, med, hi = np.percentile(null, [2.5, 50, 97.5], axis=0)
    a.fill_between(k, lo, hi, color=NULL, alpha=0.45, lw=0, label="sign-flip null (95%)")
    a.plot(k, med, color=MUTED, lw=1.2, ls="--")
    a.scatter(k, eig, color=DOWN, s=30, zorder=3, label="data")
    a.set_xticks(k)
    a.set_xlabel("rank")
    a.set_ylabel("eigenvalue of N − P")
    a.set_title("Eigenvalues of the downside-excess matrix N − P", fontsize=11)
    a.legend(fontsize=9)
    ov = abs(a_vec @ pc1) / (np.linalg.norm(a_vec) * np.linalg.norm(pc1))
    b.scatter(pc1, a_vec, s=10, color=BLUE, alpha=0.7)
    b.set_xlabel("PC1 loading (market mode of C)")
    b.set_ylabel("top eigenvector of N − P")
    b.set_title(f"…and it is the market mode (overlap {ov:.2f})", fontsize=11)
    _save(fig, path)


def plot_tails(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = df.index * 100
    ax.plot(x, df["gauss"], color=MUTED, ls="--", lw=1.4)
    ax.plot(x, df["lower"], color=DOWN, marker="o", ms=4)
    ax.plot(x, df["upper"], color=UP, marker="o", ms=4)
    xe = x[-1]
    for col, color, text in [("lower", DOWN, "joint crashes  λ_L"), ("upper", UP, "joint rallies  λ_U"),
                             ("gauss", MUTED, "Gaussian copula")]:
        ax.annotate(text, (xe, df[col].iloc[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)
    ax.set_xlim(0, xe * 1.35)
    ax.set_xlabel("tail size q (%)")
    ax.set_ylabel("P(stock i in its tail | stock j in its tail)")
    ax.set_title("Crashes are more joint than a Gaussian copula allows; rallies are not")
    _save(fig, path)


def plot_sector_heat(m: pd.DataFrame, path: Path, title: str, label: str):
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    lim = np.nanmax(np.abs(m.values))
    im = ax.imshow(m.values, cmap=DIVERGING, vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(m)), m.index)
    _clean_heatmap(ax)
    _annotate_cells(ax, m.values, ".2f", 0.0, lim * 0.6)
    cb = fig.colorbar(im, ax=ax, shrink=0.75)
    cb.set_label(label, color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title(title)
    _save(fig, path)


def plot_pc_skew(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(df))
    sig = (df["lo"] > 0) | (df["hi"] < 0)
    ax.vlines(x, df["lo"], df["hi"], color=OTHER, lw=3)
    ax.scatter(x[sig], df["skew"][sig], color=BLUE, s=40, zorder=3, label="95% CI excludes 0")
    ax.scatter(x[~sig], df["skew"][~sig], color=MUTED, s=40, zorder=3, label="not significant")
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, df.index)
    ax.set_ylabel("skewness of factor returns")
    ax.set_title("Skewness of each principal component (devolatilized, block-bootstrap 95% CI)")
    ax.legend(fontsize=9, loc="lower right")
    _save(fig, path)


def plot_cum_skew(contrib: pd.DataFrame, titles: dict[str, str], path: Path, n_label: int = 3):
    keys = list(titles)
    fig, axes = plt.subplots(len(keys), 1, figsize=(11, 1.9 * len(keys)), sharex=True)
    for ax, k in zip(axes, keys):
        c = contrib[k]
        cum = c.cumsum()
        ax.plot(cum.index, cum.values, color=BLUE, lw=1.4, drawstyle="steps-post")
        ax.axhline(0, color=INK_2, lw=0.6)
        top = sorted(c.abs().nlargest(n_label).index)
        for i, d in enumerate(top):
            ax.annotate(d.strftime("%Y-%m-%d"), (d, cum[d]), xytext=(4, [4, -12, 14][i % 3]),
                        textcoords="offset points", fontsize=8, color=INK)
            ax.plot([d], [cum[d]], "o", color=ORANGE, ms=4)
        ax.set_title(titles[k], fontsize=10)
        ax.set_ylabel("cum. skew", fontsize=8)
    fig.tight_layout()
    _save(fig, path)


def plot_component_loadings(tables: dict[str, pd.DataFrame], sectors: pd.Series, path: Path,
                            ncols: int = 3):
    colors = sector_colors(sectors)
    keys = list(tables)
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows))
    for ax, key in zip(axes.flat, keys):
        t = tables[key].iloc[::-1]
        y = np.arange(len(t))
        ax.barh(y, t["loading"], color=[colors[s] for s in t["sector"]], height=0.72,
                edgecolor=SURFACE, linewidth=1)
        ax.set_yticks(y, [f"{tk} · {si[:20]}" for tk, si in zip(t.index, t["sub_industry"])], fontsize=7.5)
        ax.axvline(0, color=INK_2, lw=0.8)
        ax.grid(axis="y", visible=False)
        ax.set_title(key, fontsize=10)
    for ax in axes.flat[len(keys):]:
        ax.axis("off")
    shown = [s for s, c in colors.items() if c != OTHER]
    folded = [s for s, c in colors.items() if c == OTHER]
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s], label=s) for s in shown]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=OTHER, label="Other: " + ", ".join(folded)))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0))
    _save(fig, path)
