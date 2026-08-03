"""AnnDictionary 式探针 Step1: 无监督聚类 → 每簇噪声 de-novo DE 基因表 + 多数真标签。
用 bcv2 conda env (sklearn/scipy)。稀疏运算控内存。
输出 new_benchmarks/annd_<tag>_clusters.json: [{cluster,n,true_label,top_genes:[...]}]
"""
import numpy as np, json, argparse, collections
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="raw_data/tenx_pbmc_cite/tenx_labeled.npz")
ap.add_argument("--label_key", default="labels7")
ap.add_argument("--k", type=int, default=15)
ap.add_argument("--top", type=int, default=30)
ap.add_argument("--tag", default="tenx")
a = ap.parse_args()

d = np.load(a.npz, allow_pickle=True)
X = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))  # cells x genes
genes = d["genes"]; labels = d[a.label_key]
print(f"loaded {X.shape} cells x genes, {len(set(labels))} true labels", flush=True)

# CPM + log1p (稀疏)
lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
Xn = X.multiply((1e4 / lib)[:, None]).tocsr(); Xn.data = np.log1p(Xn.data)

# HVG from sparse variance (E[x^2]-E[x]^2), densify only HVG for PCA
m = np.asarray(Xn.mean(0)).ravel()
m2 = np.asarray(Xn.multiply(Xn).mean(0)).ravel()
var = m2 - m * m
hvg = np.argsort(var)[-2000:]
Xhvg = Xn[:, hvg].toarray()
pca = PCA(50, random_state=0).fit_transform(Xhvg)
km = KMeans(a.k, n_init=10, random_state=0).fit_predict(pca)
print(f"clustered into {len(set(km))} clusters", flush=True)

out = []
for c in sorted(set(km)):
    mask = km == c
    if mask.sum() < 10:
        continue
    true = collections.Counter(labels[mask]).most_common(1)[0][0]
    purity = collections.Counter(labels[mask]).most_common(1)[0][1] / mask.sum()
    mi = np.asarray(Xn[mask].mean(0)).ravel()
    mr = np.asarray(Xn[~mask].mean(0)).ravel()
    score = mi - mr
    top = np.argsort(score)[-a.top:][::-1]
    out.append({"cluster": int(c), "n": int(mask.sum()), "true_label": str(true),
                "purity": round(float(purity), 2),
                "top_genes": [str(genes[i]) for i in top]})

json.dump(out, open(f"new_benchmarks/annd_{a.tag}_clusters.json", "w"), indent=1)
print(f"\n{len(out)} clusters -> new_benchmarks/annd_{a.tag}_clusters.json", flush=True)
for o in out:
    print(f"  c{o['cluster']:2d} n={o['n']:4d} pur={o['purity']:.2f} {o['true_label']:24s} {o['top_genes'][:8]}")
