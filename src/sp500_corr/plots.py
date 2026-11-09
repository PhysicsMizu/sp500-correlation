"""Figures. Static PNGs, light theme, one consistent palette."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .rmt import Spectrum, mp_pdf

# --- palette (validated categorical order; see README) ----------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
OTHER = "#c3c2b7"
BLUE, ORANGE, RED = SERIES[0], SERIES[1], SERIES[7]
DIVERGING = LinearSegmentedColormap.from_list(
    "bluered", ["#104281", "#3987e5", "#f0efec", "#e66767", "#a52a2a"]
)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def sector_colors(sectors: pd.Series) -> dict[str, str]:
    """Eight largest sectors get the eight palette slots, the rest fold to Other."""
    order = sectors.value_counts().index.tolist()
    cmap = {s: SERIES[i] for i, s in enumerate(order[:8])}
    for s in order[8:]:
        cmap[s] = OTHER
    return cmap


def plot_spectrum(spec: Spectrum, path: Path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})

    # Left: the bulk vs Marchenko-Pastur
    bulk_max = spec.lambda_plus * 2.2
    vals = spec.eigvals[spec.eigvals < bulk_max]
    a.hist(vals, bins=80, density=True, color=BLUE, alpha=0.55, edgecolor=SURFACE, linewidth=0.5,
           label="Empirical eigenvalues")
    x = np.linspace(1e-4, bulk_max, 600)
    # Density normalized to all N eigenvalues so it's comparable to the histogram
    scale = len(spec.eigvals) / len(vals)
    a.plot(x, mp_pdf(x, spec.q, spec.sigma2) * scale, color=RED,
           label=f"Marchenko-Pastur (q={spec.q:.2f}, σ²={spec.sigma2:.2f})")
    a.axvline(spec.lambda_plus, color=INK_2, lw=1, ls="--")
    a.annotate(f"λ₊ = {spec.lambda_plus:.2f}", (spec.lambda_plus, a.get_ylim()[1] * 0.5),
               xytext=(6, 0), textcoords="offset points", color=INK_2, fontsize=9)
    a.set_xlabel("eigenvalue")
    a.set_ylabel("density")
    a.set_title("Most of the spectrum is noise")
    a.legend(loc="upper right", fontsize=9)

    # Right: rank plot, log scale, shows the market mode and sector modes
    k = np.arange(1, len(spec.eigvals) + 1)
    signal = spec.eigvals > spec.lambda_plus
    b.scatter(k[~signal], spec.eigvals[~signal], s=10, color=OTHER, label="inside MP bulk")
    b.scatter(k[signal], spec.eigvals[signal], s=18, color=BLUE, label=f"signal ({signal.sum()})")
    b.axhline(spec.lambda_plus, color=INK_2, lw=1, ls="--")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlabel("rank")
    b.set_ylabel("eigenvalue")
    b.annotate(f"market mode: λ₁ = {spec.eigvals[0]:.0f}\n({spec.eigvals[0] / len(spec.eigvals):.0%} of variance)",
               (1, spec.eigvals[0]), xytext=(10, -8), textcoords="offset points",
               color=INK_2, fontsize=9, va="top")
    b.set_title("Eigenvalues by rank")
    b.legend(loc="lower left", fontsize=9)
    _save(fig, path)


def plot_ipr(spec: Spectrum, ipr: np.ndarray, path: Path):
    n = len(spec.eigvals)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    signal = spec.eigvals > spec.lambda_plus
    ax.scatter(spec.eigvals[~signal], ipr[~signal], s=10, color=OTHER, label="inside MP bulk")
    ax.scatter(spec.eigvals[signal], ipr[signal], s=18, color=BLUE, label="signal")
    ax.axhline(3 / n, color=RED, lw=1, ls="--")
    ax.annotate("random vector (3/N)", (0.01, 3 / n), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 4), textcoords="offset points", color=INK_2, fontsize=9)
    ax.axvline(spec.lambda_plus, color=INK_2, lw=1, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("eigenvalue")
    ax.set_ylabel("inverse participation ratio")
    ax.set_title("Localization: which eigenvectors concentrate on few stocks")
    ax.legend(loc="upper right", fontsize=9)
    _save(fig, path)


def plot_sector_enrichment(enrich: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    lim = np.nanmax(np.abs(enrich.values.clip(-4, 4)))
    im = ax.imshow(enrich.T.values, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(enrich.index)), enrich.index)
    ax.set_yticks(range(len(enrich.columns)), enrich.columns)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log₂(sector weight / sector share)", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("Which sectors each principal component loads on")
    _save(fig, path)


def plot_corr_matrices(raw: np.ndarray, clean: np.ndarray, sectors: pd.Series, path: Path):
    order = np.argsort(sectors.values, kind="stable")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6))
    for ax, m, title in zip(axes, [raw, clean], ["Sample correlation", "After eigenvalue clipping"]):
        im = ax.imshow(m[np.ix_(order, order)], cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    # Sector boundaries and labels on the left panel
    sorted_sec = sectors.values[order]
    edges = np.flatnonzero(sorted_sec[1:] != sorted_sec[:-1]) + 1
    bounds = np.r_[0, edges, len(sorted_sec)]
    for ax in axes:
        for e in edges:
            ax.axhline(e - 0.5, color=INK, lw=0.4, alpha=0.4)
            ax.axvline(e - 0.5, color=INK, lw=0.4, alpha=0.4)
    axes[0].set_yticks((bounds[:-1] + bounds[1:]) / 2 - 0.5, [sorted_sec[b] for b in bounds[:-1]], fontsize=8)
    cb = fig.colorbar(im, ax=axes, shrink=0.7)
    cb.outline.set_visible(False)
    _save(fig, path)


def plot_mst(g: nx.Graph, sectors: pd.Series, names: pd.Series, path: Path, title: str, label_top: int = 12):
    colors = sector_colors(sectors)
    pos = nx.kamada_kawai_layout(g, weight="weight")
    deg = dict(g.degree())
    fig, ax = plt.subplots(figsize=(13, 11))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=AXIS, width=0.8)
    nodes = list(g.nodes())
    nx.draw_networkx_nodes(
        g, pos, ax=ax, nodelist=nodes,
        node_color=[colors[sectors[n]] for n in nodes],
        node_size=[18 + 14 * deg[n] for n in nodes],
        edgecolors=SURFACE, linewidths=0.8,
    )
    top = sorted(deg, key=deg.get, reverse=True)[:label_top]
    for n in top:
        x, y = pos[n]
        ax.annotate(n, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9,
                    color=INK, fontweight="bold")
    # Legend: sectors in palette order, Other last
    shown = [s for s, c in colors.items() if c != OTHER]
    folded = [s for s, c in colors.items() if c == OTHER]
    handles = [plt.Line2D([], [], marker="o", ls="", color=colors[s], markersize=8, label=s) for s in shown]
    if folded:
        handles.append(plt.Line2D([], [], marker="o", ls="", color=OTHER, markersize=8,
                                  label="Other: " + ", ".join(folded)))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=9, ncol=3)
    ax.set_title(title + "  (node size = degree; top hubs labeled)")
    ax.axis("off")
    _save(fig, path)


def plot_rolling(stats: pd.DataFrame, path: Path):
    fig, (a, b) = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True)
    a.plot(stats.index, stats["lambda1_share"] * 100, color=BLUE)
    a.set_ylabel("% of variance")
    a.set_title("Market mode: largest eigenvalue / N  (1-year rolling window)")
    b.plot(stats.index, stats["tree_length"], color=ORANGE)
    b.set_ylabel("mean MST edge length")
    b.set_title("Normalized tree length: the tree shrinks when everything moves together")
    _save(fig, path)


def plot_gmv(oos: pd.DataFrame, path: Path):
    vol = oos.std() * np.sqrt(252) * 100
    labels = {"sample": "Sample\ncorrelation", "clipped": "RMT-clipped\ncorrelation", "equal": "Equal\nweight"}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bars = ax.bar([labels[k] for k in vol.index], vol.values, color=BLUE, width=0.55)
    for bar, v in zip(bars, vol.values):
        ax.annotate(f"{v:.1f}%", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", color=INK, fontsize=10)
    ax.set_ylabel("realized annualized volatility (%)")
    ax.set_title("Out-of-sample min-variance portfolio risk")
    ax.grid(axis="x", visible=False)
    _save(fig, path)
