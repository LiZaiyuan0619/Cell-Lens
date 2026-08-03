#!/usr/bin/env python3
"""Item 3 (B re-analysis, NO new API): per-CELL logistic of "F3 fixed this cell" ~ "marker visible".

Prior type-level correlation (n=7) was underpowered (r=-0.49, p=0.26). Redo at the CELL level:
 - outcome  y_i = 1 if (L0 wrong on cell i) AND (F3 right on cell i)   ["F3 fixed it"]
 - predictor x_i = 1 if the cell's discriminative RNA marker is VISIBLE in its top-K cell-sentence
 - covariate (2-var story from item B): per-class L0 baseline accuracy (headroom control)
Pool the per-cell outcomes of 3 deepseek/gpt/gemini mimitou CITE files (distinct models = independent
trials; same-cell-across-models is fine, same-cell-same-model duplicates are de-duped by file choice).
Logistic fit via scipy MLE (Wald p); also a 2x2 odds ratio + Fisher exact for robustness.
Also: marker-burial% per type for ALL 3 CITE sources (mimitou/10x/hao), extending the mimitou-only ledger.

Usage: .venv/bin/python analyze_burial_logit.py
"""
import json, os
import numpy as np
from scipy import sparse, stats
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs")
KG = 200  # top-K cell-sentence length used by the L0/F3 harness

# 7-class type -> discriminative RNA marker (cell-sentence visibility decides "buried or not")
TYPE2MARKER = {
    "Naive CD4+ T": "CD4", "Effector CD4+ T": "CD4",
    "Naive CD8+ T": "CD8A", "Effector CD8+ T": "CD8A",
    "B": "MS4A1", "NK": "NKG7", "Monocytes": "CD14",
}

# de-duped: one file per model (largest, non-overlapping). Same cells across aligned/non-aligned for a
# given model => keep only the bigger non-aligned file. Distinct models = independent trials.
MODEL_FILES = [
    ("deepseek_v4_flash", "results_cite/cite_deepseek_v4_flash_n60.json"),
    ("gpt54_mini",        "results_cite/cite_gpt54_mini_n50.json"),
    ("gemini35_flash",    "results_cite/cite_gemini35_flash_n50.json"),
]

CITE_SOURCES = [
    ("mimitou", "raw_data/mimitou_cite/cite_labeled.npz"),
    ("10x",     "raw_data/tenx_pbmc_cite/tenx_labeled.npz"),
    ("hao",     "raw_data/hao2021_cite/hao_labeled.npz"),
]


def load_npz(path):
    d = np.load(os.path.join(HERE, path), allow_pickle=True)
    G = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    return G, d["genes"], d["labels7"]


def topk_visible(G, genes, i, marker, kg=KG):
    """is `marker` among the top-kg expressed genes of cell i (the cell-sentence the LLM sees)?"""
    gi = np.where(genes == marker)[0]
    if len(gi) == 0:
        return None  # marker gene not in panel
    gi = gi[0]
    row = G[i].tocoo()
    if row.nnz == 0:
        return 0
    order = np.argsort(-row.data)[:kg]
    top_cols = set(int(row.col[j]) for j in order)
    return int(gi in top_cols)


# ---------- logistic via scipy MLE (Wald SE from inverse Hessian) ----------
def logit_fit(X, y):
    """X: (n,p) design WITHOUT intercept; returns betas (incl intercept), SEs, z, p (two-sided Wald)."""
    n, p = X.shape
    Xd = np.hstack([np.ones((n, 1)), X])  # add intercept

    def negll(b):
        z = Xd @ b
        # stable log(1+exp(z))
        ll = np.sum(y * z - np.logaddexp(0, z))
        return -ll

    def grad(b):
        z = Xd @ b
        mu = 1 / (1 + np.exp(-z))
        return -Xd.T @ (y - mu)

    b0 = np.zeros(p + 1)
    res = minimize(negll, b0, jac=grad, method="BFGS")
    b = res.x
    z = Xd @ b
    mu = 1 / (1 + np.exp(-z))
    W = mu * (1 - mu)
    H = Xd.T @ (Xd * W[:, None])  # Fisher information
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full_like(b, np.nan)
    zstat = b / se
    pval = 2 * (1 - stats.norm.cdf(np.abs(zstat)))
    return b, se, zstat, pval, res


def main():
    os.makedirs(FIGS, exist_ok=True)
    print("=" * 90)
    print("ITEM 3 (B re-analysis): per-CELL logistic  P(F3 fixed) ~ marker-visible (+ L0 headroom)")
    print("=" * 90)

    # mimitou npz drives marker visibility for the result files (they all sample mimitou cells)
    Gm, genes_m, labels_m = load_npz("raw_data/mimitou_cite/cite_labeled.npz")

    rows = []  # per (cell, model) observation
    per_model = {}
    for mname, fp in MODEL_FILES:
        o = json.load(open(os.path.join(HERE, fp)))
        preds, labs = o["preds"], o["labels"]
        ks = sorted(labs.keys(), key=int)
        nfix = nbroke = nvis = 0
        # per-class L0 acc (headroom covariate)
        cls_ids = {}
        for k in ks:
            cls_ids.setdefault(labs[k], []).append(k)
        l0acc = {}
        for c, ids in cls_ids.items():
            l0acc[c] = np.mean([preds.get(f"{k}|l0") == labs[k] for k in ids])
        for k in ks:
            t = labs[k]
            if t not in TYPE2MARKER:
                continue
            marker = TYPE2MARKER[t]
            vis = topk_visible(Gm, genes_m, int(k), marker)
            if vis is None:
                continue
            l0_ok = int(preds.get(f"{k}|l0") == t)
            f3_ok = int(preds.get(f"{k}|f3") == t)
            fixed = int(l0_ok == 0 and f3_ok == 1)
            broke = int(l0_ok == 1 and f3_ok == 0)
            nfix += fixed; nbroke += broke; nvis += vis
            rows.append(dict(model=mname, cell=int(k), type=t, marker=marker,
                             vis=vis, l0_ok=l0_ok, f3_ok=f3_ok, fixed=fixed, broke=broke,
                             l0base=l0acc[t]))
        per_model[mname] = (len([r for r in rows if r["model"] == mname]), nfix, nbroke, nvis)
        nn = per_model[mname][0]
        print(f"  {mname:18s}: n_cells={nn:4d}  F3-fixed={nfix:3d}  F3-broke={nbroke:3d}  marker-visible={nvis:3d}")

    n = len(rows)
    y = np.array([r["fixed"] for r in rows], float)
    x_vis = np.array([r["vis"] for r in rows], float)
    x_base = np.array([r["l0base"] for r in rows], float)
    print(f"\nPooled per-cell observations: n={n}  (F3-fixed rate={y.mean()*100:.1f}%, "
          f"marker-visible rate={x_vis.mean()*100:.1f}%)")

    # ---- 2x2 odds ratio (marker buried vs visible) + Fisher exact ----
    # rows: marker buried (vis=0) / visible (vis=1); cols: F3-fixed yes/no
    a = int(((x_vis == 0) & (y == 1)).sum())  # buried & fixed
    b = int(((x_vis == 0) & (y == 0)).sum())  # buried & not
    c = int(((x_vis == 1) & (y == 1)).sum())  # visible & fixed
    dd = int(((x_vis == 1) & (y == 0)).sum())  # visible & not
    table = [[a, b], [c, dd]]
    orr, fp_p = stats.fisher_exact(table, alternative="two-sided")
    fix_buried = a / (a + b) if (a + b) else float("nan")
    fix_visible = c / (c + dd) if (c + dd) else float("nan")
    print("\n--- 2x2: P(F3-fixed | marker buried) vs P(F3-fixed | marker visible) ---")
    print(f"  buried (vis=0):  fixed={a:4d} / {a+b:4d} = {fix_buried*100:5.1f}%")
    print(f"  visible(vis=1):  fixed={c:4d} / {c+dd:4d} = {fix_visible*100:5.1f}%")
    print(f"  Fisher odds ratio (buried-fixes-more) = {orr:.2f}   p={fp_p:.3g}")

    # ---- logistic: fixed ~ buried(=1-vis) [+ (1 - L0 baseline) headroom] ----
    x_buried = 1 - x_vis
    # model 1: marker-burial only
    b1, se1, z1, p1, _ = logit_fit(x_buried.reshape(-1, 1), y)
    or1 = np.exp(b1[1]); ci1 = (np.exp(b1[1] - 1.96 * se1[1]), np.exp(b1[1] + 1.96 * se1[1]))
    print("\n--- Logistic 1: P(F3-fixed) ~ marker_buried ---")
    print(f"  beta(buried)={b1[1]:+.3f}  SE={se1[1]:.3f}  OR={or1:.2f} [95%CI {ci1[0]:.2f}-{ci1[1]:.2f}]  "
          f"Wald z={z1[1]:.2f}  p={p1[1]:.3g}")
    # model 2: + L0 headroom (= 1 - per-class L0 baseline)
    x_room = 1 - x_base
    X2 = np.column_stack([x_buried, x_room])
    b2, se2, z2, p2, _ = logit_fit(X2, y)
    print("\n--- Logistic 2: P(F3-fixed) ~ marker_buried + L0_headroom(1-L0base) ---")
    for nm, idx in [("buried", 1), ("L0_headroom", 2)]:
        orv = np.exp(b2[idx]); ci = (np.exp(b2[idx] - 1.96 * se2[idx]), np.exp(b2[idx] + 1.96 * se2[idx]))
        print(f"  {nm:12s} beta={b2[idx]:+.3f} SE={se2[idx]:.3f} OR={orv:.2f} "
              f"[{ci[0]:.2f}-{ci[1]:.2f}] z={z2[idx]:.2f} p={p2[idx]:.3g}")

    # ---- Monocytes caveat: how much of F3-broke is Monocytes (L0 already nails) ----
    mono = [r for r in rows if r["type"] == "Monocytes"]
    mono_broke = sum(r["broke"] for r in mono)
    mono_l0 = np.mean([r["l0_ok"] for r in mono]) if mono else float("nan")
    all_broke = sum(r["broke"] for r in rows)
    print(f"\n--- Monocytes caveat ---")
    print(f"  Monocytes: n={len(mono)}  L0 acc={mono_l0*100:.1f}%  F3-broke={mono_broke} "
          f"(={100*mono_broke/max(all_broke,1):.0f}% of all {all_broke} F3-breaks)")
    print(f"  -> CellCode 'hurts' concentrate where L0 already wins (marker visible / high L0 acc).")

    # ---- cross-source marker-burial% (extends mimitou-only ledger) ----
    print("\n" + "=" * 90)
    print("Marker-burial% per type, ALL 3 CITE sources (visible% in top-200 cell-sentence)")
    print("=" * 90)
    burial = {}
    for sname, spath in CITE_SOURCES:
        try:
            G, genes, labels = load_npz(spath)
        except Exception as e:
            print(f"  {sname}: load ERR {e}"); continue
        print(f"\n[{sname}]  ({spath})")
        burial[sname] = {}
        for t, mk in TYPE2MARKER.items():
            ids = [i for i in range(len(labels)) if labels[i] == t]
            if not ids:
                print(f"  {t:16s} marker={mk:6s}  (no cells)"); continue
            vis = [topk_visible(G, genes, i, mk) for i in ids]
            vis = [v for v in vis if v is not None]
            if not vis:
                print(f"  {t:16s} marker={mk:6s}  (marker not in panel)"); continue
            pv = 100 * np.mean(vis)
            burial[sname][t] = pv
            print(f"  {t:16s} marker={mk:6s}  visible={pv:5.1f}%  (n={len(vis)})")

    # ---- figure: 2x2 bar + per-source burial heat ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.bar(["marker\nburied", "marker\nvisible"], [fix_buried * 100, fix_visible * 100],
           color=["#d62728", "#2ca02c"], width=0.6)
    for xi, v in enumerate([fix_buried * 100, fix_visible * 100]):
        ax.annotate(f"{v:.1f}%", (xi, v + 1), ha="center", fontsize=11)
    ax.set_ylabel("P(F3 fixes this cell)  [%]")
    ax.set_title(f"Per-cell: CellCode fixes buried-marker cells\nOR={orr:.1f}, Fisher p={fp_p:.1e}  (n={n})")
    ax.set_ylim(0, max(fix_buried, fix_visible) * 100 * 1.25 + 5)

    ax = axes[1]
    types = list(TYPE2MARKER.keys())
    srcs = [s for s, _ in CITE_SOURCES if s in burial]
    M = np.array([[burial[s].get(t, np.nan) for t in types] for s in srcs])
    im = ax.imshow(M, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([f"{t}\n({TYPE2MARKER[t]})" for t in types], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(srcs))); ax.set_yticklabels(srcs)
    for ii in range(len(srcs)):
        for jj in range(len(types)):
            if not np.isnan(M[ii, jj]):
                ax.text(jj, ii, f"{M[ii,jj]:.0f}", ha="center", va="center", fontsize=8)
    ax.set_title("Marker-visible% in top-200 cell-sentence\n(low=buried -> F3 helps)")
    fig.colorbar(im, ax=ax, fraction=0.046, label="visible %")
    fig.tight_layout()
    outp = os.path.join(FIGS, "burial_logit.png")
    fig.savefig(outp, dpi=150); plt.close(fig)
    print(f"\nsaved {outp}")

    json.dump({"n": n, "or_fisher": orr, "p_fisher": fp_p,
               "fix_buried": fix_buried, "fix_visible": fix_visible,
               "logit1_OR": or1, "logit1_p": p1[1],
               "logit2_buried_OR": float(np.exp(b2[1])), "logit2_buried_p": p2[1],
               "logit2_room_OR": float(np.exp(b2[2])), "logit2_room_p": p2[2],
               "mono_broke": mono_broke, "all_broke": all_broke, "mono_l0": mono_l0,
               "burial_by_source": burial},
              open(os.path.join(HERE, "results_cite/burial_logit_summary.json"), "w"), indent=1)
    print("saved results_cite/burial_logit_summary.json")


if __name__ == "__main__":
    main()
