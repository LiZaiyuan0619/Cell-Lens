"""T3: per-cluster mean surface-protein block for the AnnDictionary de-novo CTA F3 arm (flavor 1).

Reproduces the DETERMINISTIC clustering from annd_prep.py on a CITE npz (which carries surface
proteins in P_clr), then for each cluster emits the cluster-mean CLR protein signature (top-Kp).
Writes <out> keyed by cluster id. Also re-derives cluster/n/true_label/top_genes and VERIFIES they
match the existing clusters json, so the f3 protein blocks align to exactly the same clusters the
L0/cellcode/shuffle arms use.

Cluster membership is unsupervised (KMeans), so averaging protein over members does NOT leak the
true label -> a legitimate flavor-1 (restore orthogonal modality) probe on independent CTA.

Run with the bcv2 conda env (sklearn/scipy). NO API calls.
Examples:
  # tenx (14 proteins)
  python annd_protein_block.py
  # mimitou (227 proteins)
  python annd_protein_block.py --npz raw_data/mimitou_cite/cite_aligned.npz --label_key labels \
      --clusters new_benchmarks/annd_mimitou_clusters.json --out new_benchmarks/annd_mimitou_protein.json
"""
import numpy as np, json, collections, argparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="raw_data/tenx_pbmc_cite/tenx_labeled.npz")
ap.add_argument("--label_key", default="labels7")
ap.add_argument("--clusters", default="new_benchmarks/annd_tenx_clusters.json")
ap.add_argument("--out", default="new_benchmarks/annd_tenx_protein.json")
ap.add_argument("--k", type=int, default=15, help="KMeans clusters (match annd_prep --k)")
ap.add_argument("--top", type=int, default=30, help="top DE genes (match annd_prep --top, for verify)")
ap.add_argument("--kp", type=int, default=8, help="top surface proteins in the block")
ap.add_argument("--diff", action="store_true",
                help="select proteins by enrichment (cluster-mean minus global-mean); "
                     "needed for large panels with housekeeping proteins (e.g. 227-plex)")
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=True)
X = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))  # cells x genes
genes = d["genes"]; labels = d[a.label_key]; proteins = d["proteins"]; Pclr = d["P_clr"]
print(f"loaded {X.shape} cells x genes; proteins={len(proteins)}; label_key={a.label_key}", flush=True)

# --- identical preprocessing + clustering as annd_prep.py (all random_state=0 -> deterministic) ---
lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
Xn = X.multiply((1e4 / lib)[:, None]).tocsr(); Xn.data = np.log1p(Xn.data)
m = np.asarray(Xn.mean(0)).ravel()
m2 = np.asarray(Xn.multiply(Xn).mean(0)).ravel()
var = m2 - m * m
hvg = np.argsort(var)[-2000:]
Xhvg = Xn[:, hvg].toarray()
pca = PCA(50, random_state=0).fit_transform(Xhvg)
km = KMeans(a.k, n_init=10, random_state=0).fit_predict(pca)

Pglobal = Pclr.mean(0)  # for --diff enrichment ranking
out = {}; verify = []
for c in sorted(set(km)):
    mask = km == c
    if mask.sum() < 10:
        continue
    true = collections.Counter(labels[mask]).most_common(1)[0][0]
    purity = collections.Counter(labels[mask]).most_common(1)[0][1] / mask.sum()
    mi = np.asarray(Xn[mask].mean(0)).ravel(); mr = np.asarray(Xn[~mask].mean(0)).ravel()
    score = mi - mr; top = np.argsort(score)[-a.top:][::-1]
    mp = Pclr[mask].mean(0)                       # cluster-mean surface protein (CLR)
    rank_by = (mp - Pglobal) if a.diff else mp    # --diff: enrichment vs global -> discriminative
    cand = [j for j in np.argsort(-rank_by) if mp[j] > 0]   # keep positive-abundance proteins only
    porder = (cand or list(np.argsort(-rank_by)))[:a.kp]    # top-Kp (fallback if none positive)
    pblock = ", ".join(f"{proteins[j]}:{mp[j]:.1f}" for j in porder)
    out[str(int(c))] = {"protein_block": pblock, "n": int(mask.sum()), "true_label": str(true)}
    verify.append({"cluster": int(c), "n": int(mask.sum()), "true_label": str(true),
                   "purity": round(float(purity), 2), "top_genes": [str(genes[i]) for i in top]})

json.dump(out, open(a.out, "w"), indent=1)

# --- verify alignment with the clusters the other arms already use ---
exist = json.load(open(a.clusters))
match = (len(exist) == len(verify) and all(
    e["cluster"] == v["cluster"] and e["n"] == v["n"] and e["true_label"] == v["true_label"]
    and e["top_genes"][:8] == v["top_genes"][:8]
    for e, v in zip(exist, verify)))
print(f"\nMATCH existing clusters: {match}  (n_exist={len(exist)} n_new={len(verify)})", flush=True)
print(f"wrote {a.out} ({len(out)} clusters)", flush=True)
for c, v in list(out.items())[:8]:
    print(f"  c{c:>2} true={v['true_label']:22s} P: {v['protein_block']}")
if not match:
    print("\n[WARN] clustering did not reproduce the existing json exactly (sklearn drift?)."
          " Do NOT pair these protein blocks with the existing L0/cellcode/shuffle results.")
