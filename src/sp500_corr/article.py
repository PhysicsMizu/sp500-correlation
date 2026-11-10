"""Analysis behind docs/what-the-pcs-mean.md. Run: uv run sp500-article"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import data, interpret, pca, rmt
from . import plots_article as pa

ROOT = Path(__file__).resolve().parents[2]


def _pc_title(v: pd.Series, meta: pd.DataFrame, k: int) -> str:
    pos = interpret.name_factor(v, meta)
    neg = interpret.name_factor(-v, meta)
    return f"PC{k + 1}:  + {pos}   vs   − {neg}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--k", type=int, default=12, help="number of PCs to interpret")
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "article")
    args = p.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    K = args.k

    meta = data.get_constituents().set_index("ticker")
    rets = data.log_returns(data.get_prices(meta.index.tolist(), args.start, args.end))
    meta = meta.loc[rets.columns]
    T, N = rets.shape
    corr = np.corrcoef(rets.values, rowvar=False)
    spec = rmt.spectrum(corr, T)
    V = spec.eigvecs

    # 1. Who loads on it?
    titles = {f"PC{k + 1}": _pc_title(pd.Series(V[:, k], index=rets.columns), meta, k) for k in range(K)}
    show = ["PC2", "PC3", "PC4", "PC6"]
    tables = {s: interpret.top_loadings(V, meta, int(s[2:]) - 1, n=8) for s in show}
    pa.plot_loadings(tables, meta["sector"], titles, out / "loadings.png")
    with open(out / "top_loadings.md", "w") as fh:
        for k in range(K):
            t = interpret.top_loadings(V, meta, k, n=8)
            fh.write(f"### {titles[f'PC{k + 1}']}  (λ = {spec.eigvals[k]:.1f})\n\n")
            fh.write(t.assign(loading=t.loading.round(3)).to_markdown() + "\n\n")

    # 2. What does it trade like?
    pc_rets = pca.eigenportfolio_returns(rets, V, k=K)
    factors = data.get_factor_returns(args.start, args.end)
    fc = interpret.factor_correlations(pc_rets, factors)
    pa.plot_factor_corr(fc, out / "factor_correlations.png")
    pa.plot_cumulative(pc_rets, {k: titles[k] for k in show}, out / "cumulative.png")

    # 3. How much does it explain?
    r2 = interpret.r2_decomposition(spec.eigvals, V, spec.n_signal)
    r2.index = rets.columns
    pa.plot_r2_by_sector(r2, meta["sector"], out / "r2_by_sector.png")

    # 4. Is it there every year?
    act = interpret.yearly_activity(rets, V, spec.eigvals, K)
    pa.plot_activity(act, out / "activity.png")

    # 5. Is it a stable object? Raw PCs vs varimax factors, first vs second half
    a, b = interpret.split_halves(rets)
    sa = rmt.spectrum(np.corrcoef(a.values, rowvar=False), len(a))
    sb = rmt.spectrum(np.corrcoef(b.values, rowvar=False), len(b))
    raw_ov = interpret.eigvec_overlap(sa.eigvecs[:, :K], sb.eigvecs[:, :K])
    ra = interpret.rotated_factors(np.corrcoef(a.values, rowvar=False), len(a), K)
    rb = interpret.rotated_factors(np.corrcoef(b.values, rowvar=False), len(b), K)
    rot_ov = interpret.eigvec_overlap(ra, rb)
    rows, cols = linear_sum_assignment(-rot_ov)
    rot_ov = rot_ov[:, cols]
    rot_labels = [interpret.name_factor(pd.Series(ra[:, j], index=rets.columns), meta)[:22] for j in rows]
    pa.plot_stability(raw_ov, rot_ov, rot_labels, out / "stability.png")
    subspace = {k: round(interpret.subspace_overlap(sa.eigvecs[:, :k], sb.eigvecs[:, :k]), 3)
                for k in (1, 3, 5, 10, 20)}

    # Full-sample varimax factors, for the article's table
    rf = interpret.rotated_factors(corr, T, K)
    rot_table = []
    for j in range(rf.shape[1]):
        v = pd.Series(rf[:, j], index=rets.columns)
        rot_table.append({
            "factor": interpret.name_factor(v, meta),
            "variance": round(float((v**2).sum()), 1),
            "top": ", ".join(v.nlargest(6).index),
        })

    numbers = {
        "T": T, "N": N, "start": str(rets.index[0].date()), "end": str(rets.index[-1].date()),
        "halves_split": str(b.index[0].date()),
        "n_signal": spec.n_signal, "n_signal_halves": [sa.n_signal, sb.n_signal],
        "eigvals": [round(float(x), 2) for x in spec.eigvals[:K]],
        "titles": titles,
        "raw_overlap_diag": np.round(np.diag(raw_ov), 2).tolist(),
        "rot_overlap_diag": np.round(np.diag(rot_ov), 2).tolist(),
        "rot_labels": rot_labels,
        "subspace_overlap": subspace,
        "random_vector_overlap": round(float(np.sqrt(2 / np.pi / N)), 3),
        "r2_median": r2.median().round(3).to_dict(),
        "r2_by_sector": r2.groupby(meta["sector"].values).mean().round(3).to_dict("index"),
        "most_idiosyncratic": r2.nsmallest(8, "signal").round(2).to_dict("index"),
        "most_market": r2.nlargest(8, "market").round(2).to_dict("index"),
        "factor_corr_max": {pc: [fc[pc].abs().idxmax(), round(float(fc[pc][fc[pc].abs().idxmax()]), 2)]
                            for pc in fc.columns},
        "activity_max": {pc: [int(act[pc].idxmax()), round(float(act[pc].max()), 1)] for pc in act.columns},
        "varimax": rot_table,
    }
    (out / "numbers.json").write_text(json.dumps(numbers, indent=2, default=str))
    fc.round(2).to_csv(out / "factor_correlations.csv")
    act.round(2).to_csv(out / "activity.csv")
    print(json.dumps(numbers, indent=2, default=str))


if __name__ == "__main__":
    main()
