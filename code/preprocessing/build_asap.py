#!/usr/bin/env python3
"""配对 ASAP 底座: gene-activity(ATAC→基因, TF-IDF归一surface细胞特异基因) + ADT蛋白 → 标注7类 → asap_labeled.npz。
QC: 只留 total-activity>2000 的真细胞(~4660); 余为空液滴。蛋白 panel 同 CITE 227-plex, 复用 z-score 门控标注。
存的 G = TF-IDF 加权稀疏(harness 按它排序出 cell-sentence, 避免长基因偏置)。"""
import gzip, numpy as np, collections
from scipy import sparse
D="raw_data/mimitou_asap"
QC_MIN=2000

ga=np.load(f"{D}/asap_geneactivity.npz", allow_pickle=True)
G=sparse.csr_matrix((ga["data"],ga["indices"],ga["indptr"]),shape=tuple(ga["shape"]))
genes=ga["genes"]; ga_bc=np.array(ga["barcodes"])
tot=np.asarray(G.sum(1)).ravel()
keepc=np.where(tot>QC_MIN)[0]
G=G[keepc]; ga_bc=ga_bc[keepc]
print(f"QC tot>{QC_MIN}: {len(keepc)} cells (from {len(tot)})", flush=True)
bc_pos={b:i for i,b in enumerate(ga_bc)}

# ADT 对齐
ADT=f"{D}/GSM4732110_CD28_CD3_control_ASAP_ADT.tsv.gz"
with gzip.open(ADT) as f:
    prot_names=np.array(f.readline().decode().rstrip("\n").split("\t")[1:])
    P=np.zeros((len(ga_bc),len(prot_names)),dtype=np.float32); seen=np.zeros(len(ga_bc),bool)
    for line in f:
        tab=line.find(b"\t"); i=bc_pos.get(line[:tab].decode())
        if i is None: continue
        P[i]=np.array(line.decode().rstrip("\n").split("\t")[1:],dtype=np.float32); seen[i]=True
print(f"ADT matched {seen.sum()}/{len(ga_bc)}", flush=True)
keep=np.where(seen)[0]; G=G[keep]; P=P[keep]; barcodes=ga_bc[keep]
N=G.shape[0]

# TF-IDF on gene-activity → surface 细胞特异基因 (标准 scATAC)
df=np.asarray((G>0).sum(0)).ravel()
idf=np.log((N+1)/(df+1))+1
rowsum=np.asarray(G.sum(1)).ravel()+1e-9
Gc=G.tocoo()
tf=Gc.data/rowsum[Gc.row]
tfidf=sparse.csr_matrix((tf*idf[Gc.col],(Gc.row,Gc.col)),shape=G.shape)

# 蛋白标注 (同 CITE)
def clr(x):
    x=x+1.0; g=np.exp(np.mean(np.log(x),1,keepdims=True)); return np.log(x/g)
Pc=clr(P); prot=list(prot_names)
z=(Pc-Pc.mean(0))/(Pc.std(0)+1e-9)
def Z(*names):
    cols=[prot.index(x) for x in names if x in prot]; return z[:,cols].max(1) if cols else np.full(N,-9.0)
CD3=Z("CD3.1","CD3.2");CD4=Z("CD4.1","CD4.2");CD8=Z("CD8","CD8a");CD19=Z("CD19")
CD14=Z("CD14");CD56=Z("CD56.NCAM.","CD56.NCAM.Recombinant","CD335");CD45RA=Z("CD45RA");CD45RO=Z("CD45RO")
labels7=[]
for i in range(N):
    if CD19[i]>1.0 and CD3[i]<0.5: labels7.append("B"); continue
    if CD14[i]>1.0 and CD3[i]<0.5: labels7.append("Monocytes"); continue
    if CD56[i]>1.0 and CD3[i]<0.5: labels7.append("NK"); continue
    if CD3[i]>0.0:
        arm="CD8" if CD8[i]>CD4[i] else "CD4"; state="Naive" if CD45RA[i]>CD45RO[i] else "Effector"
        labels7.append(f"{state} {arm}+ T"); continue
    labels7.append("Unassigned")
labels7=np.array(labels7)
print("=== ASAP labels7 composition ===");
for c,n in collections.Counter(labels7).most_common(): print(f"  {c:18s} {n}")

# sanity: T 细胞 vs B 细胞的 TF-IDF top 基因
def topgenes(mask,k=15):
    idx=np.where(mask)[0]
    if len(idx)==0: return []
    sub=tfidf[idx]; mean=np.asarray(sub.mean(0)).ravel()
    return [genes[j] for j in np.argsort(-mean)[:k]]
print("\nT(Naive CD4) top TF-IDF genes:", " ".join(topgenes(np.char.startswith(labels7,"Naive CD4"))))
print("B top TF-IDF genes:", " ".join(topgenes(labels7=="B")))
print("Mono top TF-IDF genes:", " ".join(topgenes(labels7=="Monocytes")))

np.savez_compressed(f"{D}/asap_labeled.npz",
    data=tfidf.data,indices=tfidf.indices,indptr=tfidf.indptr,shape=np.array(tfidf.shape),
    genes=genes,proteins=prot_names,P_raw=P,P_clr=Pc,
    barcodes=barcodes,labels7=labels7,labels_rna=labels7)
print(f"\nsaved {D}/asap_labeled.npz")
