#!/usr/bin/env python3
"""ASAP-seq ATAC fragments → gene-activity 矩阵 (cell × gene)。
gene activity = 落在 (gene body + 2kb 上游 promoter) 区间内的 fragment 计数之和 (Signac 式)。
只保留有蛋白(ADT)的细胞。用 ncls 做快速区间重叠。设计在 rjob 大内存节点跑。
用法: python compute_gene_activity.py
"""
import gzip, numpy as np, collections, sys, time
from scipy import sparse
try:
    from ncls import NCLS
except ImportError:
    print("need ncls: pip install ncls", file=sys.stderr); raise

D="raw_data/mimitou_asap"
FRAG=f"{D}/GSM4732109_CD28_CD3_control_ASAP_fragments.tsv.gz"
ADT=f"{D}/GSM4732110_CD28_CD3_control_ASAP_ADT.tsv.gz"
BED="raw_data/annotation/genes_hg38.bed"
PROMO=2000

t0=time.time()
# 1) 保留细胞 = 有蛋白的 barcode
print("load ADT barcodes...", flush=True)
with gzip.open(ADT) as f:
    f.readline()  # header
    adt_bc=set(line[:line.find(b"\t")].decode() for line in f)
print(f"  ADT barcodes: {len(adt_bc)}", flush=True)

# 2) 基因区间 (protein_coding + lncRNA), 按 chrom 建 NCLS
print("load genes...", flush=True)
genes=[]; chrom_iv=collections.defaultdict(lambda:[[],[],[]])  # chrom -> [starts,ends,ids]
for ln in open(BED):
    c,s,e,strand,name,typ=ln.rstrip("\n").split("\t")
    if typ not in ("protein_coding","lncRNA"): continue
    s=int(s); e=int(e); gi=len(genes); genes.append(name)
    if strand=="+": rs,re=max(0,s-PROMO),e
    else: rs,re=s,e+PROMO
    chrom_iv[c][0].append(rs); chrom_iv[c][1].append(re); chrom_iv[c][2].append(gi)
genes=np.array(genes)
ncls_by={}
for c,(ss,ee,ii) in chrom_iv.items():
    ncls_by[c]=NCLS(np.array(ss,dtype=np.int64),np.array(ee,dtype=np.int64),np.array(ii,dtype=np.int64))
print(f"  genes: {len(genes)}, chroms: {len(ncls_by)}", flush=True)

# 3) 流式读 fragments, 重叠基因 → 累加 (cell,gene)
print("stream fragments...", flush=True)
bc_idx={}  # barcode -> cell index (只对保留细胞编号)
rows=[]; cols=[]; vals=[]
n=0; kept=0
with gzip.open(FRAG) as f:
    for line in f:
        if line.startswith(b"#"): continue
        n+=1
        if n%5_000_000==0: print(f"  {n//1_000_000}M frags [{time.time()-t0:.0f}s] cells={len(bc_idx)}", flush=True)
        p=line.rstrip(b"\n").split(b"\t")
        if len(p)<4: continue
        bc=p[3].decode()
        if bc not in adt_bc: continue
        chrom=p[0].decode();
        nc=ncls_by.get(chrom)
        if nc is None: continue
        s=int(p[1]); e=int(p[2]); cnt=int(p[4]) if len(p)>4 else 1
        ci=bc_idx.get(bc)
        if ci is None: ci=len(bc_idx); bc_idx[bc]=ci
        # 重叠 [s,e)
        for _,_,gi in nc.find_overlap(s,e):
            rows.append(ci); cols.append(gi); vals.append(cnt)
        kept+=1
print(f"  total frags={n}, kept(in-cell)={kept}, cells={len(bc_idx)} [{time.time()-t0:.0f}s]", flush=True)

ncell=len(bc_idx)
M=sparse.coo_matrix((vals,(rows,cols)),shape=(ncell,len(genes))).tocsr()
M.sum_duplicates()
barcodes=np.array(sorted(bc_idx,key=lambda b:bc_idx[b]))
print(f"gene-activity matrix: {M.shape}, nnz={M.nnz}", flush=True)
np.savez_compressed(f"{D}/asap_geneactivity.npz",
    data=M.data,indices=M.indices,indptr=M.indptr,shape=np.array(M.shape),
    genes=genes,barcodes=barcodes)
print(f"saved {D}/asap_geneactivity.npz [{time.time()-t0:.0f}s]", flush=True)
