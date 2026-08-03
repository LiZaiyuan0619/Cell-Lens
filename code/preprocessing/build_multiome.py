#!/usr/bin/env python3
"""配对 10x Multiome (RNA + ATAC) 底座 → 自标注 PBMC 谱系 → 计算 ATAC gene-activity → multiome_labeled.npz。

数据: raw_data/multiome_10k/pbmc10k_multiome.h5 (CellRanger-ARC, 已 filtered, GRCh38)
  - feature_type "Gene Expression" (36601 基因, symbol in features/name) + "Peaks" (108377, interval=chrN:start-end)
QC: rna_lib>1000 & n_genes>500 (几乎全留 = 已是 cellranger filtered matrix)
自标注 (caveat: 这是 kill-test 的弱标签, 非金标准): RNA 标记基因 log-normalized score → argmax 入 6 谱系
  CD4T (IL7R/CD4/CCR7), CD8T (CD8A/CD8B), B (MS4A1/CD79A/CD79B), NK (NKG7/GNLY/KLRD1),
  Mono (CD14/LYZ/FCN1/FCGR3A), (DC CD3 阴, 可选). 先算 T(CD3D/E/G) 门控再分 CD4/CD8。
ATAC gene-activity (Signac 式): 落在 gene body + 2kb 上游 promoter 的 peak 计数之和 (NCLS 区间重叠)。
  peak 坐标直接来自 features/interval, 无需 GTF; 基因区间用 raw_data/annotation/genes_hg38.bed。
存的 G = RNA raw counts (harness 按它排序出 cell-sentence); GA = ATAC gene-activity (cells x genes_ga, 用作恢复的模态块)。
"""
import h5py, numpy as np, collections, time
from scipy import sparse
from ncls import NCLS

H5="raw_data/multiome_10k/pbmc10k_multiome.h5"
BED="raw_data/annotation/genes_hg38.bed"
PROMO=2000
RNA_LIB_MIN=1000; NGENE_MIN=500
t0=time.time()

# ── 1) load h5, split RNA / ATAC ──
f=h5py.File(H5,"r"); g=f["matrix"]
ftype=np.array([t.decode() for t in g["features"]["feature_type"][:]])
fname=np.array([t.decode() for t in g["features"]["name"][:]])
fival=np.array([t.decode() for t in g["features"]["interval"][:]])
M=sparse.csc_matrix((g["data"][:],g["indices"][:],g["indptr"][:]),shape=tuple(g["shape"][:]))  # features x cells
barcodes=np.array([b.decode() for b in g["barcodes"][:]])
Mt=M.T.tocsr()  # cells x features
rna_mask=ftype=="Gene Expression"; atac_mask=ftype=="Peaks"
RNA=Mt[:,rna_mask].tocsr(); ATAC=Mt[:,atac_mask].tocsr()
rna_genes=fname[rna_mask]; peak_ival=fival[atac_mask]
print(f"[{time.time()-t0:.0f}s] RNA {RNA.shape} ATAC {ATAC.shape}", flush=True)

# ── 2) QC ──
rna_lib=np.asarray(RNA.sum(1)).ravel(); rna_ng=np.asarray((RNA>0).sum(1)).ravel()
keep=np.where((rna_lib>RNA_LIB_MIN)&(rna_ng>NGENE_MIN))[0]
RNA=RNA[keep]; ATAC=ATAC[keep]; barcodes=barcodes[keep]
N=RNA.shape[0]; print(f"QC kept {N} cells", flush=True)

# ── 3) 自标注: RNA marker log-norm score → argmax ──
gidx={}
for j,gn in enumerate(rna_genes):
    gidx.setdefault(gn,j)  # 重复 symbol 取首个
libsz=np.asarray(RNA.sum(1)).ravel()+1.0
def score(*gs):
    s=np.zeros(N)
    for gn in gs:
        if gn in gidx:
            col=np.asarray(RNA[:,gidx[gn]].todense()).ravel()
            s=s+np.log1p(col/libsz*1e4)
    return s
S_T   =score("CD3D","CD3E","CD3G","TRAC")
S_CD4 =score("IL7R","CD4","CCR7")
S_CD8 =score("CD8A","CD8B")
S_B   =score("MS4A1","CD79A","CD79B")
S_NK  =score("NKG7","GNLY","KLRD1","NCAM1")
S_Mono=score("CD14","LYZ","FCN1","FCGR3A")
labels=[]
for i in range(N):
    big={"T":S_T[i],"B":S_B[i],"NK":S_NK[i],"Monocyte":S_Mono[i]}
    top=max(big,key=big.get)
    if top=="T":
        labels.append("CD8 T" if S_CD8[i]>S_CD4[i] else "CD4 T")
    else:
        labels.append(top)
labels=np.array(labels)
# broad labels (T 不分 CD4/CD8) for leakage-robust option
labels_broad=np.array(["T" if l in ("CD4 T","CD8 T") else l for l in labels])
print("=== self-annotated labels (RNA marker argmax) ===")
for c,n in collections.Counter(labels).most_common(): print(f"  {c:12s} {n}")

# ── 4) ATAC gene-activity: peak interval → gene (body + 2kb promoter) via NCLS ──
print(f"[{time.time()-t0:.0f}s] build gene NCLS...", flush=True)
ga_genes=[]; chrom_iv=collections.defaultdict(lambda:[[],[],[]])
for ln in open(BED):
    c,s,e,strand,name,typ=ln.rstrip("\n").split("\t")
    if typ not in ("protein_coding","lncRNA"): continue
    s=int(s); e=int(e); gi=len(ga_genes); ga_genes.append(name)
    if strand=="+": rs,re=max(0,s-PROMO),e
    else: rs,re=s,e+PROMO
    chrom_iv[c][0].append(rs); chrom_iv[c][1].append(re); chrom_iv[c][2].append(gi)
ga_genes=np.array(ga_genes)
ncls_by={c:NCLS(np.array(ss,dtype=np.int64),np.array(ee,dtype=np.int64),np.array(ii,dtype=np.int64))
         for c,(ss,ee,ii) in chrom_iv.items()}
print(f"  GA genes {len(ga_genes)}, chroms {len(ncls_by)}", flush=True)

# parse each peak interval → list of overlapping gene indices
print(f"[{time.time()-t0:.0f}s] map peaks->genes...", flush=True)
peak2genes=[]  # for each ATAC col, list of GA gene idx
n_unmapped=0
for ival in peak_ival:
    try:
        chrom,rng=ival.split(":"); s,e=rng.split("-"); s=int(s); e=int(e)
    except Exception:
        peak2genes.append([]); n_unmapped+=1; continue
    nc=ncls_by.get(chrom)
    if nc is None: peak2genes.append([]); n_unmapped+=1; continue
    hits=[gi for _,_,gi in nc.find_overlap(s,e)]
    peak2genes.append(hits)
mapped_peaks=sum(1 for h in peak2genes if h)
print(f"  peaks mapped to >=1 gene: {mapped_peaks}/{len(peak_ival)} (unparsable/no-chrom {n_unmapped})", flush=True)

# build peak->gene sparse projection (peaks x ga_genes), then GA = ATAC @ proj  (cells x ga_genes)
prows=[]; pcols=[]
for pj,hits in enumerate(peak2genes):
    for gi in hits: prows.append(pj); pcols.append(gi)
proj=sparse.csr_matrix((np.ones(len(prows),dtype=np.float32),(prows,pcols)),
                       shape=(len(peak_ival),len(ga_genes)))
GA=(ATAC.astype(np.float32) @ proj).tocsr()  # cells x ga_genes (gene-activity counts)
print(f"[{time.time()-t0:.0f}s] GA {GA.shape} nnz={GA.nnz}", flush=True)
ga_per_cell=np.asarray((GA>0).sum(1)).ravel()
print(f"  GA genes/cell quantiles: {np.percentile(ga_per_cell,[10,50,90]).astype(int)}", flush=True)

# sanity: top GA genes per lineage (do they look ATAC-marker-like?)
def top_ga(mask,k=12):
    idx=np.where(mask)[0]
    if not len(idx): return []
    m=np.asarray(GA[idx].mean(0)).ravel()
    return [ga_genes[j] for j in np.argsort(-m)[:k]]
print("\nT  top GA genes:", " ".join(top_ga(labels_broad=="T")))
print("B  top GA genes:", " ".join(top_ga(labels=="B")))
print("Mono top GA genes:", " ".join(top_ga(labels=="Monocyte")))

# ── 5) save npz (schema for harness) ──
out="raw_data/multiome_10k/multiome_labeled.npz"
np.savez_compressed(out,
    data=RNA.data,indices=RNA.indices,indptr=RNA.indptr,shape=np.array(RNA.shape),
    genes=rna_genes,
    ga_data=GA.data,ga_indices=GA.indices,ga_indptr=GA.indptr,ga_shape=np.array(GA.shape),
    ga_genes=ga_genes,
    barcodes=barcodes,labels=labels,labels_broad=labels_broad)
print(f"\nsaved {out} [{time.time()-t0:.0f}s]")
