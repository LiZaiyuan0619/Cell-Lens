#!/usr/bin/env python3
"""跨数据集#3: Hao 2021 (GSE164378) 3' PBMC CITE-seq → 与 Mimitou/10x 同 schema 底座。
RNA(GSM5008737, 161764×33538) + ADT(GSM5008738, 228-plex) 已逐 barcode 1:1 对齐(同序)。
去同型对照 → CLR → z-score 门控标注 7 类 → 每类下采样(cap) → 存 hao_labeled.npz。
RNA 矩阵大(161k 细胞), 走 rjob 大内存节点。
"""
import numpy as np, gzip, collections, os, subprocess
from scipy import io as sio, sparse
D="raw_data/hao2021_cite"
TAR=f"{D}/GSE164378_RAW.tar"
CAP=1500  # 每类最多保留
need=["GSM5008737_RNA_3P-matrix.mtx.gz","GSM5008737_RNA_3P-features.tsv.gz",
      "GSM5008738_ADT_3P-matrix.mtx.gz","GSM5008738_ADT_3P-features.tsv.gz"]
for f in need:
    if not os.path.exists(f"{D}/{f}"):
        print("extract",f,flush=True); subprocess.run(["tar","-xf",TAR,"-C",D,f],check=True)

# ---- ADT (small): CLR + 门控标注 ----
adt_names=[l.decode().rstrip("\n").split("\t")[1] for l in gzip.open(f"{D}/GSM5008738_ADT_3P-features.tsv.gz")]
print("loading ADT matrix...",flush=True)
ADT=np.asarray(sio.mmread(f"{D}/GSM5008738_ADT_3P-matrix.mtx.gz").todense(),dtype=np.float64).T  # cells x prot
keep=np.array([not any(s in p.lower() for s in ["igg","isotype","control"]) for p in adt_names])
ADT=ADT[:,keep]; proteins=np.array(adt_names)[keep]
n=ADT.shape[0]; prot=list(proteins)
print(f"cells={n} proteins(kept)={len(proteins)}",flush=True)
L=np.log1p(ADT); P_clr=(L-L.mean(1,keepdims=True))
z=(P_clr-P_clr.mean(0))/(P_clr.std(0)+1e-9)
def Z(*nm):
    cols=[prot.index(x) for x in nm if x in prot]
    if not cols: print("  WARN marker missing:",nm,flush=True)
    return z[:,cols].max(1) if cols else np.full(n,-9.0)
CD3=Z("CD3-1","CD3-2","CD3"); CD4=Z("CD4-1","CD4-2"); CD8=Z("CD8","CD8a")
CD19=Z("CD19"); CD56=Z("CD56-1","CD56-2","CD56"); CD14=Z("CD14"); CD45RA=Z("CD45RA"); CD45RO=Z("CD45RO")
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
print("=== labels7 (full) ===",flush=True)
for c,m in collections.Counter(labels7).most_common(): print(f"  {c:18s} {m}")

# ---- 下采样平衡 ----
CLASSES=["Naive CD4+ T","Effector CD4+ T","Naive CD8+ T","Effector CD8+ T","B","NK","Monocytes"]
rng=np.random.default_rng(0); sel=[]
for c in CLASSES:
    ids=np.where(labels7==c)[0]; rng.shuffle(ids); sel+=list(ids[:CAP])
sel=np.array(sorted(sel)); print(f"\nsubsampled cells={len(sel)}",flush=True)

# ---- RNA (大): 只取选中细胞列 ----
genes=np.array([l.decode().rstrip("\n").split("\t")[1] for l in gzip.open(f"{D}/GSM5008737_RNA_3P-features.tsv.gz")])
print("loading RNA matrix (large)...",flush=True)
RNA=sio.mmread(f"{D}/GSM5008737_RNA_3P-matrix.mtx.gz").tocsc()      # genes x cells
GEX=RNA[:,sel].T.tocsr(); GEX.data=GEX.data.astype(np.int64)         # sel-cells x genes
P_sel=P_clr[sel]; lab_sel=labels7[sel]

print("\n=== sanity: mean CLR by labels7 (subset) ===",flush=True)
mk={"CD3":["CD3-1","CD3-2","CD3"],"CD4":["CD4-1","CD4-2"],"CD8":["CD8","CD8a"],"CD45RA":["CD45RA"],"CD45RO":["CD45RO"],"CD19":["CD19"],"CD14":["CD14"],"CD56":["CD56-1","CD56-2"]}
hdr=" ".join(f"{m:>6}" for m in mk); print(f"{'type':18s} n    {hdr}")
for lab in CLASSES:
    mask=lab_sel==lab; nn=mask.sum()
    if nn==0: print(f"{lab:18s} 0"); continue
    vals=[P_sel[mask][:,[prot.index(c) for c in cols if c in prot]].max(1).mean() for m,cols in mk.items()]
    print(f"{lab:18s} {nn:4d} "+" ".join(f"{v:6.2f}" for v in vals))

np.savez_compressed(f"{D}/hao_labeled.npz",
    data=GEX.data,indices=GEX.indices,indptr=GEX.indptr,shape=GEX.shape,
    genes=genes,proteins=proteins,P_clr=P_sel,labels7=lab_sel,labels_rna=lab_sel)
print(f"\nsaved {D}/hao_labeled.npz  shape={GEX.shape}",flush=True)
