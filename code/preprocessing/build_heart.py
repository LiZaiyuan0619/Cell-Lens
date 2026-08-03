"""Build heart (GSE217494) CITE → npz for run_cite_ab.py (--labelset heart).
3rd non-PBMC battlefield: SOLID ORGAN (heart) — has STRUCTURAL cell types absent from
blood/marrow (Endothelium/Fibroblast/SMC-Pericyte/Endocardium). Strongest breadth axis.
Schema matches run_cite_ab.py: GEX raw-count CSR + genes + proteins + P_clr(CLR) + labels_init.

Layout: 22 per-sample 10x MTX (features×cells, GEX+ADT interleaved via feature_types);
metadata bc = "<core>-<sample>", per-sample barcodes.tsv = "<core>-1". Join by (sample, core).
Memory-safe: subsample ≤K cells/class, then per-sample mmread + column-subset (free each iter).
"""
import numpy as np, gzip, collections, pandas as pd
from scipy import sparse
from scipy.io import mmread

BASE = "raw_data/heart_gse217494"
K = 300
NAT = {"Endothelium": "Endothelial cell", "Fibroblast": "Fibroblast", "Myeloid": "Myeloid cell",
       "SMC_Pericyte": "Smooth muscle cell", "T_NK Cells": "T cell", "Endocardium": "Endocardial cell",
       "B Cells": "B cell"}
KEEP = set(NAT)   # 7 classes with enough cells + unambiguous names (drop Glia/Lymphatics/Mast/Plasma <400)

# ---------- 1. metadata → subsample ≤K per class, group by sample ----------
meta = pd.read_csv(f"{BASE}/GSE217494_cell.metadata.csv.gz")
print("=== celltype dist (all) ===")
for c, n in collections.Counter(meta["celltype"]).most_common():
    print(f"  {c:16s} {n}")
meta = meta[meta["celltype"].isin(KEEP)].copy()
rng = np.random.default_rng(0)
keep_idx = []
for c in sorted(KEEP):
    idx = meta.index[meta["celltype"] == c].to_numpy()
    rng.shuffle(idx)
    keep_idx.extend(idx[:K].tolist())
sel = meta.loc[keep_idx]
targets = {}   # sample -> {core_barcode: raw_celltype}
for bc, samp, ct in zip(sel["bc"], sel["sample"], sel["celltype"]):
    core = bc.rsplit("-", 1)[0]
    targets.setdefault(samp, {})[core] = ct
print(f"\nselected {len(sel)} cells across {len(targets)} samples; "
      f"per-class = { {c:int((sel['celltype']==c).sum()) for c in sorted(KEEP)} }")

# ---------- 2. per-sample load matrix (features×cells), subset target cols ----------
GEX_parts, ADT_parts, labels = [], [], []
genes = proteins = None
for samp, tmap in targets.items():
    ft = pd.read_csv(f"{BASE}/GSE217494_{samp}.features.tsv.gz", sep="\t", header=None)
    ftype = ft[2].values
    gm = ftype == "Gene Expression"; am = ftype == "Antibody Capture"
    if genes is None:
        genes = ft[1].values[gm].astype(object); proteins = ft[1].values[am].astype(object)
    bcs = pd.read_csv(f"{BASE}/GSE217494_{samp}.barcodes.tsv.gz", header=None)[0].values
    core = np.array([b.rsplit("-", 1)[0] for b in bcs], object)
    col_idx = [i for i, cb in enumerate(core) if cb in tmap]
    if not col_idx:
        continue
    M = mmread(gzip.open(f"{BASE}/GSE217494_{samp}.matrix.mtx.gz", "rt")).tocsc()  # feat×cells
    sub = M[:, col_idx]
    GEX_parts.append(sub[gm, :].T.tocsr())                      # cells×genes (raw counts)
    ADT_parts.append(np.asarray(sub[am, :].todense(), np.float64).T)   # cells×279
    labels.extend(NAT[tmap[core[i]]] for i in col_idx)
    print(f"  {samp}: {len(col_idx)} cells", flush=True)

GEX = sparse.vstack(GEX_parts).tocsr()
ADT = np.vstack(ADT_parts)
labels = np.array(labels, object)

def clr(Mx):
    Mx = Mx + 1.0
    L = np.log(Mx)
    return L - L.mean(axis=1, keepdims=True)
P_clr = clr(ADT)

print(f"\nGEX {GEX.shape}  ADT {ADT.shape}  proteins(head) {list(proteins[:10])}")
print("=== final label dist ===")
for c, n in collections.Counter(labels).most_common():
    print(f"  {c:26s} {n}")
np.savez(f"{BASE}/heart_labeled.npz",
         data=GEX.data, indices=GEX.indices, indptr=GEX.indptr, shape=np.array(GEX.shape),
         genes=genes, proteins=proteins, P_clr=P_clr, labels_init=labels)
print(f"saved {BASE}/heart_labeled.npz")
