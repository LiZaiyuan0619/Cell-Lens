#!/usr/bin/env python3
"""配对 Mimitou CITE-seq control 底座: GEX(RNA, MatrixMarket稀疏) + ADT(227蛋白, dense tsv barcode-major)。
- 流式读 ADT(2.77M raw barcodes)只保留 GEX-filtered 的 5527 细胞 → 低内存。
- 蛋白 CLR 归一 + marker-规则做免疫谱系标注(reliable, 因 CITE 蛋白本就是 cell type 的金标准)。
- 同时打印 repertoire 构成 + CellVerse CITE 候选类型, 供 F3 设计参考。
输出 raw_data/mimitou_cite/cite_aligned.npz。
"""
import gzip, json, numpy as np
from scipy import sparse
from scipy.io import mmread
import io, collections

D="raw_data/mimitou_cite"
GEX_BC=f"{D}/GSM4732113_CD28_CD3_control_CITE_GEX_barcodes.tsv.gz"
GEX_FT=f"{D}/GSM4732113_CD28_CD3_control_CITE_GEX_features.tsv.gz"
GEX_MX=f"{D}/GSM4732113_CD28_CD3_control_CITE_GEX_matrix.tsv.gz"
ADT=f"{D}/GSM4732114_CD28_CD3_control_CITE_ADT.tsv.gz"

print("load GEX barcodes/features...", flush=True)
barcodes=[l.decode().strip() for l in gzip.open(GEX_BC)]
feat_rows=[l.decode().strip().split("\t") for l in gzip.open(GEX_FT)]
genes=np.array([r[1] if len(r)>1 else r[0] for r in feat_rows])  # symbol
print(f"  GEX: {len(genes)} genes x {len(barcodes)} cells", flush=True)

print("load GEX matrix (MatrixMarket)...", flush=True)
with gzip.open(GEX_MX) as f:
    M=mmread(io.BytesIO(f.read()))          # genes x cells (coordinate integer)
M=sparse.csr_matrix(M)
if M.shape[0]==len(barcodes) and M.shape[0]!=len(genes):
    M=M.T.tocsr()
print(f"  matrix shape {M.shape} (genes x cells), nnz={M.nnz}", flush=True)
Gx=M.T.tocsr()                              # cells x genes
bc_set={b:i for i,b in enumerate(barcodes)}

print("stream ADT, keep GEX cells only...", flush=True)
with gzip.open(ADT) as f:
    header=f.readline().decode().rstrip("\n").split("\t")
    prot_names=np.array(header[1:])         # 227 proteins
    P=np.zeros((len(barcodes), len(prot_names)), dtype=np.float32)
    seen=np.zeros(len(barcodes), dtype=bool)
    n=0
    for line in f:
        n+=1
        tab=line.find(b"\t")
        bc=line[:tab].decode()
        i=bc_set.get(bc)
        if i is None: continue
        vals=line.decode().rstrip("\n").split("\t")[1:]
        P[i]=np.array(vals, dtype=np.float32)
        seen[i]=True
    print(f"  scanned {n} ADT rows; matched {seen.sum()}/{len(barcodes)} GEX cells", flush=True)

# 只保留 GEX+ADT 都有的细胞
keep=np.where(seen)[0]
Gx=Gx[keep]; P=P[keep]; barcodes=[barcodes[i] for i in keep]
print(f"paired cells: {len(keep)}", flush=True)

# 蛋白 CLR 归一 (per cell across proteins)
def clr(x):
    x=x+1.0
    g=np.exp(np.mean(np.log(x),axis=1,keepdims=True))
    return np.log(x/g)
Pc=clr(P)
pidx={p:i for i,p in enumerate(prot_names)}
def pget(*names):
    for nm in names:
        if nm in pidx: return Pc[:,pidx[nm]]
    return np.full(len(keep), -1e9)

# 关键谱系蛋白
CD3=np.maximum(pget("CD3.1"),pget("CD3.2"))
CD4=np.maximum(pget("CD4.1"),pget("CD4.2"))
CD8=np.maximum(pget("CD8"),pget("CD8a"))
CD19=pget("CD19"); CD20=pget("CD20")
CD56=np.maximum(pget("CD56.NCAM.","CD56.NCAM.Recombinant"),pget("CD335"))
CD14=pget("CD14"); CD16=pget("CD16"); CD11c=pget("CD11c"); CD123=pget("CD123")
CD45RA=pget("CD45RA"); CD45RO=pget("CD45RO")

# 规则标注(蛋白金标准): 先大类, T 内再 CD4/CD8 + naive/effector
def annotate():
    labs=[]
    for i in range(len(keep)):
        if CD19[i]>0.5 or CD20[i]>0.5: labs.append("B"); continue
        if CD14[i]>0.5 and CD3[i]<0.3: labs.append("Monocyte"); continue
        if (CD11c[i]>1.0 or CD123[i]>1.0) and CD3[i]<0.3 and CD14[i]<0.5: labs.append("DC"); continue
        if CD3[i]>0.5:  # T
            if CD8[i]>CD4[i] and CD8[i]>0.3:
                base="CD8 T"
            elif CD4[i]>0.3:
                base="CD4 T"
            else:
                base="T (other)"
            if base in ("CD4 T","CD8 T"):
                state="Naive" if CD45RA[i]>CD45RO[i] else "Effector/Memory"
                labs.append(f"{state} {base}")
            else:
                labs.append(base)
            continue
        if CD56[i]>0.5 and CD3[i]<0.3: labs.append("NK"); continue
        labs.append("Unassigned")
    return np.array(labs)

labels=annotate()
print("\n=== repertoire (protein-rule labels) ===", flush=True)
for c,n in collections.Counter(labels).most_common():
    print(f"  {c:24s} {n}", flush=True)

# RNA-marker 备份标签(leakage-control 用: 与蛋白独立)
gidx={g:i for i,g in enumerate(genes)}
Gn=Gx.astype(np.float64)
libsz=np.asarray(Gn.sum(1)).ravel()+1
def rscore(*gs):
    s=np.zeros(len(keep))
    for g in gs:
        if g in gidx: s=s+np.log1p(np.asarray(Gn[:,gidx[g]].todense()).ravel()/libsz*1e4)
    return s
RNA_T=rscore("CD3D","CD3E","CD3G"); RNA_B=rscore("MS4A1","CD79A","CD79B")
RNA_NK=rscore("NKG7","GNLY","KLRD1"); RNA_Mono=rscore("CD14","LYZ","FCN1")
rna_lab=[]
for i in range(len(keep)):
    sc={"T":RNA_T[i],"B":RNA_B[i],"NK":RNA_NK[i],"Monocyte":RNA_Mono[i]}
    rna_lab.append(max(sc,key=sc.get))
rna_lab=np.array(rna_lab)
print("\n=== RNA-marker broad labels (independent of protein) ===", flush=True)
for c,n in collections.Counter(rna_lab).most_common(): print(f"  {c:12s} {n}", flush=True)

np.savez_compressed(f"{D}/cite_aligned.npz",
    data=Gx.data, indices=Gx.indices, indptr=Gx.indptr, shape=np.array(Gx.shape),
    genes=genes, proteins=prot_names, P_raw=P, P_clr=Pc,
    barcodes=np.array(barcodes), labels=labels, rna_labels=rna_lab)
print(f"\nsaved {D}/cite_aligned.npz", flush=True)

# 顺手: CellVerse CITE 候选类型 (完整)
try:
    d=json.load(open("CellVerse_repo/data/cta_cite_100.json"))
    u=d[0]["messages"][1]["content"]
    seg=u.split("candidate cell types:")[-1].split("Make your choice")[0]
    print("\n=== CellVerse CITE candidate cell types ===")
    print(seg.strip()[:600])
except Exception as e:
    print("cellverse cite parse err", e)
