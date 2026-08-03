#!/usr/bin/env python3
"""跨数据集#2: 10x 官方 PBMC TotalSeq-B CITE-seq → 与 Mimitou 同 schema 的重建底座。
读 filtered_feature_bc_matrix(features×cells, 含 Gene Expression + Antibody Capture 两类 feature),
拆 GEX(cells×genes 计数 CSR) + ADT(cells×prot CLR), 去 IgG 同型对照, z-score 门控标注 CellVerse 7 类。
产物 tenx_labeled.npz 直接喂 run_cite_ab.py --npz。
"""
import numpy as np, gzip, collections
from scipy import io as sio, sparse
D="raw_data/tenx_pbmc_cite/filtered_feature_bc_matrix"
feat=[l.decode().rstrip("\n").split("\t") for l in gzip.open(f"{D}/features.tsv.gz")]
names=[f[1] for f in feat]; types=[f[2] for f in feat]
is_gex=np.array([t=="Gene Expression" for t in types])
is_adt=np.array([t=="Antibody Capture" for t in types])
print(f"features: GEX={is_gex.sum()} ADT={is_adt.sum()}")
M=sio.mmread(f"{D}/matrix.mtx.gz").tocsr()           # features x cells
GEX=M[np.where(is_gex)[0]].T.tocsr()                  # cells x genes
GEX.data=GEX.data.astype(np.int64)
genes=np.array([names[i] for i in range(len(names)) if is_gex[i]])
adt_idx=np.where(is_adt)[0]
ADT=np.asarray(M[adt_idx].T.todense(),dtype=np.float64)   # cells x prot
prot_names=np.array([names[i].replace("_TotalSeqB","") for i in adt_idx])
# 去 IgG 同型对照(背景, 不当信号)
keep=np.array([("control" not in p.lower() and not p.lower().startswith("igg")) for p in prot_names])
ADT=ADT[:,keep]; proteins=prot_names[keep]
print(f"cells={GEX.shape[0]} genes={GEX.shape[1]} proteins(kept)={list(proteins)}")
# CLR per cell
L=np.log1p(ADT); P_clr=L-L.mean(1,keepdims=True)
n=P_clr.shape[0]; prot=list(proteins)
z=(P_clr-P_clr.mean(0))/(P_clr.std(0)+1e-9)
def Z(*nm):
    cols=[prot.index(x) for x in nm if x in prot]
    return z[:,cols].max(1) if cols else np.full(n,-9.0)
CD3=Z("CD3"); CD4=Z("CD4"); CD8=Z("CD8a"); CD19=Z("CD19"); CD56=Z("CD56")
CD14=Z("CD14"); CD45RA=Z("CD45RA"); CD45RO=Z("CD45RO")
labels7=[]
for i in range(n):
    if CD19[i]>1.0 and CD3[i]<0.5: labels7.append("B"); continue
    if CD14[i]>1.0 and CD3[i]<0.5: labels7.append("Monocytes"); continue
    if CD56[i]>1.0 and CD3[i]<0.5: labels7.append("NK"); continue
    if CD3[i]>0.0:
        arm="CD8" if CD8[i]>CD4[i] else "CD4"
        state="Naive" if CD45RA[i]>CD45RO[i] else "Effector"
        labels7.append(f"{state} {arm}+ T"); continue
    labels7.append("Unassigned")
labels7=np.array(labels7)
print("\n=== labels7 distribution ===")
for c,m in collections.Counter(labels7).most_common(): print(f"  {c:18s} {m}")
print("\n=== sanity: mean CLR by labels7 ===")
def Pc_mean(P,mask,cols): return P[mask][:,cols].max(1).mean() if cols else float('nan')
mk={"CD3":["CD3"],"CD4":["CD4"],"CD8a":["CD8a"],"CD45RA":["CD45RA"],"CD45RO":["CD45RO"],"CD19":["CD19"],"CD14":["CD14"],"CD56":["CD56"]}
hdr=" ".join(f"{m:>6}" for m in mk); print(f"{'type':18s} n    {hdr}")
for lab in ["Naive CD4+ T","Effector CD4+ T","Naive CD8+ T","Effector CD8+ T","B","NK","Monocytes"]:
    mask=labels7==lab; nn=mask.sum()
    if nn==0: print(f"{lab:18s} 0"); continue
    vals=[Pc_mean(P_clr,mask,[prot.index(c) for c in cols if c in prot]) for m,cols in mk.items()]
    print(f"{lab:18s} {nn:4d} "+" ".join(f"{v:6.2f}" for v in vals))
np.savez_compressed("raw_data/tenx_pbmc_cite/tenx_labeled.npz",
    data=GEX.data,indices=GEX.indices,indptr=GEX.indptr,shape=GEX.shape,
    genes=genes,proteins=proteins,P_clr=P_clr,labels7=labels7,labels_rna=labels7)
print("\nsaved raw_data/tenx_pbmc_cite/tenx_labeled.npz")
