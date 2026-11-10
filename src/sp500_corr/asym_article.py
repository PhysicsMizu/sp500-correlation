"""Analysis behind docs/asymmetry.md (Part II). Run: uv run sp500-asymmetry"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import asymmetry as asy
from . import data, interpret
from . import plots_asymmetry as pa

ROOT = Path(__file__).resolve().parents[2]


def _r(x, n=3):
    return round(float(x), n)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--n-null", type=int, default=200, help="sign-flip null replications")
    p.add_argument("--k", type=int, default=12, help="PCs kept for the coskewness tensor")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "figures" / "asymmetry")
    args = p.parse_args(argv)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    meta = data.get_constituents().set_index("ticker")
    rets = data.log_returns(data.get_prices(meta.index.tolist(), args.start, args.end))
    meta = meta.loc[rets.columns]
    sectors = meta["sector"].values
    raw = asy.standardize(rets)
    dev = asy.ewma_devol(rets)
    Y = {"raw": raw.values, "devol": dev.values}
    nums: dict = {"T_raw": len(raw), "T_devol": len(dev), "N": rets.shape[1]}

    # ---- 1. down vs up market correlation ---------------------------------------
    print("1/4 conditional correlations")
    qs = np.round(np.arange(0.05, 0.501, 0.05), 2)
    curves, cond = {}, {}
    for label, y in Y.items():
        ex = asy.exceedance_curve(y, qs)
        gauss = asy.exceedance_curve(asy.gaussian_twin(y, rng), qs)
        flips = [asy.exceedance_curve(asy.sign_flip(y, rng), qs) for _ in range(args.n_null // 2)]
        pooled = np.concatenate([np.stack([f["down"].values for f in flips]),
                                 np.stack([f["up"].values for f in flips])])
        band = pd.DataFrame({"lo": np.percentile(pooled, 2.5, axis=0),
                             "hi": np.percentile(pooled, 97.5, axis=0)}, index=qs)
        title = "Raw returns" if label == "raw" else "Devolatilized returns (÷ EWMA vol)"
        curves[title] = {"data": ex, "gauss": gauss, "band": band}

        s = asy.conditional_stats(y, 0.2)
        null = asy.conditional_null(y, 0.2, args.n_null // 2, rng)
        cond[label] = {
            stat: {
                "down": _r(s["down"][stat]), "up": _r(s["up"][stat]),
                "diff": _r(s["down"][stat] - s["up"][stat]),
                "null_sd": _r(null[stat].std()),
                "z": _r((s["down"][stat] - s["up"][stat]) / null[stat].std(), 1),
            }
            for stat in s["down"]
        }
        g = asy.conditional_stats(asy.gaussian_twin(y, rng), 0.2)
        cond[label]["gaussian_mean_corr"] = {"down": _r(g["down"]["mean_corr"]), "up": _r(g["up"]["mean_corr"])}
    off = ~np.eye(len(rets.columns), dtype=bool)
    cond["unconditional_mean_corr"] = _r(np.corrcoef(Y["raw"], rowvar=False)[off].mean())
    pa.plot_exceedance(curves, out / "exceedance.png")
    nums["conditional_q20"] = cond

    # ---- 2. semicovariance ----------------------------------------------------------
    print("2/4 semicovariance")
    semi = {}
    for label, y in Y.items():
        P, Nn, M = asy.semicovariance(y)
        A = Nn - P
        vals, vecs = np.linalg.eigh(A)
        vals, vecs = vals[::-1], vecs[:, ::-1]
        null_mean, null_eig = [], []
        for _ in range(args.n_null):
            An = asy.downside_excess(asy.sign_flip(y, rng))
            null_mean.append(An[off].mean())
            null_eig.append(np.linalg.eigvalsh(An)[::-1][:10])
        null_eig = np.array(null_eig)
        c = np.corrcoef(y, rowvar=False)
        cv, cV = np.linalg.eigh(c)
        pc1 = cV[:, -1] * np.sign(cV[:, -1].sum())
        a1 = vecs[:, 0] * np.sign(vecs[:, 0].sum())
        semi[label] = {
            "mean_offdiag": {"P": _r(P[off].mean(), 4), "N": _r(Nn[off].mean(), 4), "M": _r(M[off].mean(), 4),
                             "C": _r((P + Nn + M)[off].mean(), 4)},
            "A_mean_offdiag": _r(A[off].mean(), 4),
            "A_mean_z": _r(A[off].mean() / np.std(null_mean), 1),
            "A_top_eigs": np.round(vals[:5], 2).tolist(),
            "null_top_eig_mean_sd": [_r(null_eig[:, 0].mean(), 2), _r(null_eig[:, 0].std(), 2)],
            "null_eig2_p97_5": _r(np.percentile(null_eig[:, 1], 97.5), 2),
            "overlap_top_A_with_PC1": _r(abs(a1 @ pc1), 3),
        }
        if label == "devol":
            pa.plot_semicov(vals[:10], null_eig, a1, pc1, out / "semicov.png")
    nums["semicovariance"] = semi

    # ---- 3. tail dependence -----------------------------------------------------------
    print("3/4 tail dependence")
    u = asy.to_uniform(Y["devol"])
    ug = asy.gaussian_copula_twin(u, rng)
    tq = [0.01, 0.025, 0.05, 0.075, 0.1]
    rows = []
    for q in tq:
        L, U = asy.tail_dependence(u, q)
        Lg, Ug = asy.tail_dependence(ug, q)
        nd = []
        for _ in range(args.n_null // 2):
            Lf, Uf = asy.tail_dependence(asy.to_uniform(asy.sign_flip(Y["devol"], rng)), q)
            nd.append(Lf[off].mean() - Uf[off].mean())
        d = L[off].mean() - U[off].mean()
        rows.append({"q": q, "lower": L[off].mean(), "upper": U[off].mean(),
                     "gauss": (Lg[off].mean() + Ug[off].mean()) / 2, "z": d / np.std(nd)})
    tails = pd.DataFrame(rows).set_index("q")
    pa.plot_tails(tails, out / "tails.png")
    L, U = asy.tail_dependence(u, 0.05)
    tail_sec = asy.sector_block_means(L - U, sectors)
    pa.plot_sector_heat(tail_sec, out / "tail_sectors.png",
                        "Excess crash co-occurrence λ_L − λ_U by sector pair (q = 5%)", "λ_L − λ_U")
    per_stock = pd.Series((L - U).sum(axis=1) / (len(L) - 1), index=rets.columns)
    nums["tails"] = {
        "by_q": tails.round(3).reset_index().to_dict("records"),
        "within_sector_q5": {s: _r(tail_sec.loc[s, s]) for s in tail_sec.index},
        "most_crash_coupled": per_stock.nlargest(10).round(3).to_dict(),
        "least_crash_coupled": per_stock.nsmallest(10).round(3).to_dict(),
    }

    # ---- 4. coskewness tensor ---------------------------------------------------------
    print("4/4 coskewness tensor")
    K = args.k
    sk = {}
    for label, y in Y.items():
        F, V, ev = asy.whiten(y, K)
        pc_sk = []
        for j in range(K):
            bs = asy.block_bootstrap_skew(F[:, j], rng)
            pc_sk.append({"pc": f"PC{j + 1}", "skew": asy.skewness(F[:, j]),
                          "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
        pc_sk = pd.DataFrame(pc_sk).set_index("pc")
        m = y.mean(axis=1)
        bs = asy.block_bootstrap_skew(m, rng)
        sk[label] = {
            "pc_skew": pc_sk.round(2).reset_index().to_dict("records"),
            "median_stock_skew": _r(np.median([asy.skewness(y[:, i]) for i in range(y.shape[1])]), 2),
            "index_skew": _r(asy.skewness(m), 2),
            "index_skew_ci": np.round(np.percentile(bs, [2.5, 97.5]), 2).tolist(),
        }
        if label == "devol":
            pa.plot_pc_skew(pc_sk, out / "pc_skew.png")
    nums["skewness"] = sk

    y = Y["devol"]
    dates = dev.index
    F, V, ev = asy.whiten(y, K)
    S = asy.coskewness(F)
    kap, Uc = asy.tensor_power_decomposition(S, 6, rng)
    loads = V @ (np.sqrt(ev)[:, None] * Uc)  # corr(stock, component)
    comps, tables, titles = [], {}, {}
    contrib = pd.DataFrame(index=dates)
    h = len(y) // 2

    def comp_loads(yy):
        Fh, Vh, eh = asy.whiten(yy, K)
        _, Uh = asy.tensor_power_decomposition(asy.coskewness(Fh), 6, rng)
        return Vh @ (np.sqrt(eh)[:, None] * Uh)

    first, second = comp_loads(y[:h]), comp_loads(y[h:])
    ov_first = interpret.eigvec_overlap(loads, first).max(axis=1)
    ov_second = interpret.eigvec_overlap(loads, second).max(axis=1)
    ov_halves = interpret.eigvec_overlap(first, second).max(axis=1)
    rf = interpret.rotated_factors(np.corrcoef(y, rowvar=False), len(y), K)
    ov_varimax = interpret.eigvec_overlap(loads, rf).max(axis=1)
    for j in range(Uc.shape[1]):
        f = F @ Uc[:, j]
        f = (f - f.mean()) / f.std()
        c = pd.Series(f**3 / len(f), index=dates)
        key = f"C{j + 1}"
        contrib[key] = c
        ld = pd.Series(loads[:, j], index=rets.columns)
        pos, neg = interpret.name_factor(ld, meta), interpret.name_factor(-ld, meta)
        titles[key] = f"{key}: + {pos} vs − {neg}"
        tables[f"{key} (skew {c.sum():+.2f})"] = interpret.top_loadings(loads, meta, j, n=6)
        bs = asy.block_bootstrap_skew(f, rng)
        top = c.abs().nlargest(10)
        comps.append({
            "component": key, "plus": pos, "minus": neg,
            "skew": _r(c.sum(), 2), "ci": np.round(np.percentile(bs, [2.5, 97.5]), 2).tolist(),
            "top10_share": _r(c[top.index].sum() / c.sum(), 2),
            "top_days": {str(d.date()): _r(c[d] / c.sum(), 2) for d in top.index[:5]},
            "top_plus": ld.nlargest(6).index.tolist(), "top_minus": ld.nsmallest(6).index.tolist(),
            "weight_on_pcs": np.round(Uc[:, j], 2).tolist(),
            "overlap_first_half": _r(ov_first[j], 2), "overlap_second_half": _r(ov_second[j], 2),
            "overlap_varimax": _r(ov_varimax[j], 2),
        })
    nums["coskew_components"] = comps
    nums["coskew_halves_overlap"] = np.round(ov_halves, 2).tolist()
    pa.plot_cum_skew(contrib, titles, out / "coskew_events.png")
    pa.plot_component_loadings(tables, meta["sector"], out / "coskew_loadings.png")

    (out / "numbers.json").write_text(json.dumps(nums, indent=2, default=str))
    print(json.dumps(nums, indent=2, default=str))


if __name__ == "__main__":
    main()
