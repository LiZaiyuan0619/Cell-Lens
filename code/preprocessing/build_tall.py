"""Build T-ALL (GSE248287) CITE → npz for run_cite_ab.py (--labelset tall).
2nd non-PBMC breadth battlefield: T-cell acute lymphoblastic leukemia (DISEASE tissue),
adds Malignant blasts + disease context absent in healthy PBMC/BMMC.
Schema matches run_cite_ab.py: GEX raw-count CSR + genes + proteins + P_clr(CLR) + labels7(coarse).

Data layout: RNA/ADT pre-split MTX (features×cells); shared barcode order == metadata row order.
Metadata is R write.table with UNHEADERED rowname → data cols are +1 vs header; col0 = barcode.
Memory-safe: dev has ~10GB free, RNA has 273M nnz → we SUBSAMPLE ≤K cells/class and STREAM
the RNA mtx in chunks keeping only selected columns (never materialise the full 273M matrix).
"""
import numpy as np, gzip, collections
from scipy import sparse
from scipy.io import mmread
import pandas as pd

BASE = "raw_data/tall_gse248287"
K = 300          # cells per class cap (enough for run --n up to 300)
LABEL_COL = "Celltypes_all"   # coarse lineage annotation (Malignant/Tcells/Myeloid/...)

# ---------- 1. labels (offset +1 for R rowname) ----------
with gzip.open(f"{BASE}/GSE248287_10x_metadata_all.txt.gz", "rt") as f:
    hdr = f.readline().rstrip("\n").split("\t")
    off = 1
    ci = hdr.index(LABEL_COL)
    ci2 = hdr.index("Celltypes_initial")
    labels_all, labels_init = [], []
    for line in f:
        p = line.rstrip("\n").split("\t")
        labels_all.append(p[ci + off])
        labels_init.append(p[ci2 + off])
labels_all = np.array(labels_all, object)
labels_init = np.array(labels_init, object)
# map dataset's concatenated codes → NATURAL names so LLM free-text ("T cell") matches
# (parse_pred does substring match; "Tcells" would never match "T cell"). Malignant kept
# but excluded from CANDS (it's a disease STATE, not a lineage the model would ever name).
NAT = {"Tcells": "T cell", "Bcells": "B cell", "NKcells": "NK cell", "Myeloid": "Monocyte",
       "mDCs": "Dendritic cell", "pDCs": "Plasmacytoid dendritic cell",
       "HSPCs": "Hematopoietic stem cell", "Malignant": "Malignant blast"}
labels_init_nat = np.array([NAT.get(x, x) for x in labels_init], object)
N = len(labels_all)
print(f"total cells {N}")
print(f"=== {LABEL_COL} dist ===")
for c, n in collections.Counter(labels_all).most_common():
    print(f"  {c:16s} {n}")
print("=== Celltypes_initial dist ===")
for c, n in collections.Counter(labels_init).most_common():
    print(f"  {c:16s} {n}")

# ---------- 2. subsample ≤K per class → selected 0-based cell positions ----------
rng = np.random.default_rng(0)
sel = []
for c in sorted(set(labels_all)):
    idx = np.where(labels_all == c)[0]
    rng.shuffle(idx)
    sel.extend(idx[:K].tolist())
sel = np.array(sorted(sel))                       # 0-based cell positions
labels_sel = labels_all[sel]
sel_1based = set((sel + 1).tolist())              # MTX columns are 1-based
col_remap = {old1: new for new, old1 in enumerate(sel + 1)}   # old 1-based col → new 0-based
print(f"\nselected {len(sel)} cells; per-class = "
      f"{ {c: int((labels_sel==c).sum()) for c in sorted(set(labels_sel))} }")

# ---------- 3. stream RNA mtx (genes×cells) keeping only selected cols → cells×genes ----------
def read_mtx_selected(path, keep_1based, remap, n_new):
    with gzip.open(path, "rt") as f:
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()
        ngenes, ncells, nnz = map(int, line.split())
        rows, cols, vals = [], [], []
        reader = pd.read_csv(f, sep=" ", header=None, names=["r", "c", "v"],
                             dtype={"r": np.int32, "c": np.int32, "v": np.float32},
                             chunksize=8_000_000)
        seen = 0
        for ch in reader:
            seen += len(ch)
            sub = ch[ch["c"].isin(keep_1based)]
            if len(sub):
                rows.append(sub["r"].to_numpy() - 1)               # gene 0-based
                cols.append(sub["c"].map(remap).to_numpy().astype(np.int32))
                vals.append(sub["v"].to_numpy())
        print(f"  streamed {seen} nnz from {path.split('/')[-1]}")
    R = np.concatenate(rows); C = np.concatenate(cols); V = np.concatenate(vals)
    M = sparse.coo_matrix((V, (C, R)), shape=(n_new, ngenes)).tocsr()  # cells×genes
    return M, ngenes

GEX, ngenes = read_mtx_selected(f"{BASE}/GSE248287_10x_RNA_matrix.mtx.gz",
                                sel_1based, col_remap, len(sel))
with gzip.open(f"{BASE}/GSE248287_10x_RNA_features.tsv.gz", "rt") as f:
    genes = np.array([l.rstrip("\n").split("\t")[-1] for l in f], object)
assert len(genes) == ngenes, (len(genes), ngenes)

# ---------- 4. ADT (small: 20M nnz) via mmread, subset selected cols → CLR ----------
ADTm = mmread(gzip.open(f"{BASE}/GSE248287_10x_ADT_matrix.mtx.gz", "rt")).tocsc()  # 204×cells
ADT = np.asarray(ADTm[:, sel].todense(), dtype=np.float64).T                        # cells×204
with gzip.open(f"{BASE}/GSE248287_10x_ADT_features.tsv.gz", "rt") as f:
    proteins = np.array([l.rstrip("\n").split("\t")[-1] for l in f], object)

def clr(M):
    M = M + 1.0
    L = np.log(M)
    return L - L.mean(axis=1, keepdims=True)
P_clr = clr(ADT)

print(f"\nGEX {GEX.shape}  ADT {ADT.shape}  proteins(head) {list(proteins[:10])}")
np.savez(f"{BASE}/tall_labeled.npz",
         data=GEX.data, indices=GEX.indices, indptr=GEX.indptr, shape=np.array(GEX.shape),
         genes=genes, proteins=proteins, P_clr=P_clr, labels7=labels_sel,
         labels_init=labels_init_nat[sel])
print(f"saved {BASE}/tall_labeled.npz")
