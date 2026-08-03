"""Build BMMC (GSE194122 NeurIPS'21 CITE) → npz for run_cite_ab.py (--labelset bmmc).
Non-PBMC breadth battlefield: bone marrow adds Erythroid + HSPC/progenitor lineages absent in PBMC.
Schema matches run_cite_ab.py: GEX raw-count CSR (data/indices/indptr/shape) + genes + proteins + P_clr(CLR) + labels7(coarse).
GEX/ADT split by var['feature_types']; raw counts from layers['counts'] (X is normalized).
"""
import numpy as np, anndata as ad, collections
from scipy import sparse

A = ad.read_h5ad("raw_data/bmmc_cite/GSE194122_cite_BMMC.h5ad")
ft = A.var["feature_types"].astype(str).values
gex_m = ft == "GEX"; adt_m = ft == "ADT"
counts = sparse.csr_matrix(A.layers["counts"])            # cells × 14087, raw integer counts

GEX = counts[:, gex_m].tocsr()                            # cells × 13953 raw counts
genes = np.asarray(A.var_names[gex_m], dtype=object)
ADT = np.asarray(counts[:, adt_m].todense(), dtype=np.float64)   # cells × 134 raw
proteins = np.asarray(A.var_names[adt_m], dtype=object)

# CLR per cell (standard CITE-seq ADT normalization), matching build_tenx pipeline intent
def clr(M):
    M = M + 1.0
    L = np.log(M)
    return L - L.mean(axis=1, keepdims=True)
P_clr = clr(ADT)

# 45 fine cell_type → 11 coarse lineage classes (★ = non-PBMC).  Unmapped → OTHER (excluded by CANDS).
COARSE = {}
def m(names, lab):
    for n in names: COARSE[n] = lab
m(["CD14+ Mono"], "CD14+ Mono")
m(["CD16+ Mono"], "CD16+ Mono")
m(["CD4+ T activated","CD4+ T naive","CD4+ T activated integrinB7+","CD4+ T CD314+ CD45RA+","T reg"], "CD4 T")
m(["CD8+ T naive","CD8+ T CD57+ CD45RO+","CD8+ T CD57+ CD45RA+","CD8+ T TIGIT+ CD45RO+","CD8+ T TIGIT+ CD45RA+",
   "CD8+ T CD49f+","CD8+ T CD69+ CD45RO+","CD8+ T CD69+ CD45RA+","CD8+ T naive CD127+ CD26- CD101-"], "CD8 T")
m(["NK","NK CD158e1+"], "NK")
m(["Naive CD20+ B IGKC+","Naive CD20+ B IGKC-","Transitional B","B1 B IGKC+","B1 B IGKC-"], "B")
m(["Plasma cell IGKC+","Plasma cell IGKC-","Plasmablast IGKC+","Plasmablast IGKC-"], "Plasma")
m(["Reticulocyte","Erythroblast","Proerythroblast","Normoblast"], "Erythroid")           # ★ non-PBMC
m(["HSC","G/M prog","Lymph prog","MK/E prog","T prog cycling"], "HSPC/Prog")              # ★ non-PBMC
m(["cDC2","cDC1"], "cDC")
m(["pDC"], "pDC")
# left OTHER: MAIT, gdT*, dnT, ILC, ILC1 (ambiguous innate/gamma-delta) — excluded from task

ct = A.obs["cell_type"].astype(str).values
labels7 = np.array([COARSE.get(c, "OTHER") for c in ct], dtype=object)

print("shape GEX", GEX.shape, "ADT", ADT.shape)
print("proteins(head):", list(proteins[:12]))
print("=== coarse label distribution ===")
for c, n in collections.Counter(labels7).most_common():
    print(f"  {c:14s} {n}")

np.savez("raw_data/bmmc_cite/bmmc_labeled.npz",
         data=GEX.data, indices=GEX.indices, indptr=GEX.indptr, shape=np.array(GEX.shape),
         genes=genes, proteins=proteins, P_clr=P_clr, labels7=labels7)
print("\nsaved raw_data/bmmc_cite/bmmc_labeled.npz")
