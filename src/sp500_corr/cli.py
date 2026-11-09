"""End-to-end pipeline: data -> correlation -> RMT cleaning -> PCA -> MST -> figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, mst, pca, plots, portfolio, rmt

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    p.add_argument("--n-pcs", type=int, default=10, help="PCs to show in the sector heatmap")
    p.add_argument("--out", type=Path, default=ROOT / "figures")
    args = p.parse_args(argv)
    out: Path = args.out

    print("1/6  constituents + prices")
    meta = data.get_constituents(refresh=args.refresh).set_index("ticker")
    prices = data.get_prices(meta.index.tolist(), args.start, args.end, refresh=args.refresh)
    rets = data.log_returns(prices)
    meta = meta.loc[rets.columns]
    sectors, names = meta["sector"], meta["name"]
    T, N = rets.shape
    print(f"     {N} stocks x {T} days ({rets.index[0].date()} to {rets.index[-1].date()})")

    print("2/6  correlation + Marchenko-Pastur")
    corr = np.corrcoef(rets.values, rowvar=False)
    clean, spec = rmt.clip(corr, n_obs=T)
    plots.plot_spectrum(spec, out / "spectrum.png")
    plots.plot_corr_matrices(corr, clean, sectors, out / "correlation_matrices.png")

    print("3/6  principal components")
    ipr = pca.inverse_participation_ratio(spec.eigvecs)
    plots.plot_ipr(spec, ipr, out / "ipr.png")
    enrich = pca.sector_enrichment(spec.eigvecs, sectors, k=args.n_pcs)
    plots.plot_sector_enrichment(enrich, out / "pc_sectors.png")
    pc_rets = pca.eigenportfolio_returns(rets, spec.eigvecs, k=5)
    ew = rets.mean(axis=1)
    pc1_vs_market = float(np.corrcoef(pc_rets["PC1"], ew)[0, 1])
    # Most over-represented sector per PC (raw weight would just favor big sectors)
    top_sector = enrich.idxmax(axis=1)

    print("4/6  minimum spanning trees")
    g_raw = mst.build_mst(corr, list(rets.columns))
    g_detoned = mst.build_mst(rmt.detone(clean), list(rets.columns))
    plots.plot_mst(g_raw, sectors, names, out / "mst.png", "MST of sample correlations")
    plots.plot_mst(g_detoned, sectors, names, out / "mst_detoned.png",
                   "MST after clipping + removing the market mode")
    purity_raw = mst.sector_purity(g_raw, sectors)
    purity_det = mst.sector_purity(g_detoned, sectors)

    print("5/6  rolling market mode + tree length")
    roll = mst.rolling_stats(rets)
    plots.plot_rolling(roll, out / "rolling.png")

    print("6/6  out-of-sample min-variance test")
    oos = portfolio.rolling_gmv_backtest(rets)
    plots.plot_gmv(oos, out / "gmv_oos.png")
    vol = (oos.std() * np.sqrt(252)).round(4).to_dict()

    summary = {
        "n_stocks": N,
        "n_days": T,
        "q": round(spec.q, 4),
        "sigma2_noise": round(spec.sigma2, 4),
        "lambda_plus": round(spec.lambda_plus, 4),
        "n_signal_eigenvalues": spec.n_signal,
        "lambda1": round(float(spec.eigvals[0]), 2),
        "lambda1_share": round(float(spec.eigvals[0] / N), 4),
        "variance_share_signal": round(float(spec.eigvals[: spec.n_signal].sum() / N), 4),
        "pc1_corr_with_equal_weight_index": round(pc1_vs_market, 4),
        "pc_top_sector": top_sector.to_dict(),
        "mst_hubs": mst.hubs(g_raw).to_dict(),
        "mst_hubs_detoned": mst.hubs(g_detoned).to_dict(),
        "mst_sector_purity": purity_raw,
        "mst_sector_purity_detoned": purity_det,
        "tree_length_min": {str(roll["tree_length"].idxmin().date()): round(float(roll["tree_length"].min()), 4)},
        "gmv_oos_annual_vol": vol,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nFigures written to {out}")


if __name__ == "__main__":
    main()
