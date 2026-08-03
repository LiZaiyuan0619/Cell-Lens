#!/usr/bin/env python3
"""F3 第三模态 kill-test (10x Multiome RNA+ATAC)。改编自 run_cite_ab.py。
命题: cell-sentence (RNA top-K) 丢弃了第二测量模态; CITE 丢的是蛋白(F3 已证), ASAP 丢的是染色质衍生 gene-sentence。
本测试: 把 ATAC chromatin accessibility (gene-activity) 作为"恢复的模态块"接到 prompt → CTA 是否提升?
臂:
 l0          = 纯 RNA cell-sentence (top-K 基因, 降序) [复现基线]
 l0spec      = l0 + RNA 说明
 f3atac      = l0 + 本细胞 top-K ATAC gene-activity 基因块 (TF-IDF 排序, NAME:v=log1p activity) [诚实版: 按模态原样恢复]
 f3atacshuf  = l0 + 随机另一细胞的 ATAC 块 (跨细胞置换控制)
 f3panel     = l0 + 固定谱系 marker 的 ATAC gene-activity 值 (chromatin 的 marker panel, 蛋白 panel 的染色质类比) [最强版]
 f3panelshuf = l0 + 随机另一细胞的 panel 块
用法: .venv/bin/python multiome_ab.py --model deepseek_v4_flash --n 150 --kg 200 --kp 30 \
        --arms l0,f3atac,f3atacshuf,f3panel,f3panelshuf --workers 16 --base_url endpoint3
"""
import os, json, re, time, argparse, yaml, collections
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

NPZ="raw_data/multiome_10k/multiome_labeled.npz"
SYS="You are an expert who knows a lot about single cell biology and genomics and will help me solve a series of tasks related to single cell data analysis."

SPEC_RNA=("Note on the gene list: genes are listed in descending order of expression (most expressed first); "
          "only the top expressed genes are shown and a gene's absence does not imply it is biologically off.")
SPEC_ATAC=("Note on chromatin accessibility: this is a 10x Multiome cell, so besides RNA it was also measured for "
           "chromatin accessibility (ATAC-seq). ATAC peak counts were aggregated into per-gene 'gene-activity' scores "
           "(reads in each gene's body + promoter). Below the genes, the most accessible genes are listed as NAME:v "
           "(v = log(1+gene-activity), higher = more open chromatin at that gene), in descending order. Open chromatin "
           "marks genes that are poised/active in this cell's lineage. Use it together with the RNA genes.")
SPEC_PANEL=("Note on chromatin accessibility: this is a 10x Multiome cell with ATAC-seq. For a fixed panel of canonical "
            "lineage-defining genes, the chromatin gene-activity (log(1+ATAC reads in gene body+promoter)) is listed as "
            "NAME:v. Higher v = more open chromatin at that lineage gene. Use this chromatin readout with the RNA genes.")

# 染色质 marker panel (蛋白 panel 的类比): 经典 PBMC 谱系基因
PANEL_GENES=["CD3D","CD3E","CD3G","TRAC","CD8A","CD8B","CD4","IL7R","CCR7","SELL","TCF7",
             "MS4A1","CD79A","CD79B","CD19","NKG7","GNLY","KLRD1","NCAM1","GZMB","GZMK",
             "CD14","LYZ","FCN1","FCGR3A","S100A8","S100A9"]

EPS={"endpoint1":"https://api.example.com/v1","endpoint2":"https://api.example.com/v1",
     "endpoint3":"https://api.example.com/v1"}
def load_model(name, base_url=""):
    y=yaml.safe_load(open("AtomWorldBench/src/config/models.yaml"))[name]
    key=os.environ.get(y["api_key"].strip("${}"))
    url=EPS.get(base_url, base_url) or os.environ.get("LLM_BASE_URL") or y["base_url"]
    return OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or url, timeout=120, max_retries=2), y["model_name"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="deepseek_v4_flash")
    ap.add_argument("--n",type=int,default=150,help="每类最多取多少细胞")
    ap.add_argument("--kg",type=int,default=200,help="RNA top基因数")
    ap.add_argument("--kp",type=int,default=30,help="ATAC top gene-activity 数")
    ap.add_argument("--arms",default="l0,f3atac,f3atacshuf,f3panel,f3panelshuf")
    ap.add_argument("--workers",type=int,default=16)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--max_tokens",type=int,default=2048)
    ap.add_argument("--out",default="results_multiome")
    ap.add_argument("--labelset",default="fine",choices=["fine","broad"],
                    help="fine=CD4T/CD8T/B/NK/Mono; broad=T/B/NK/Mono")
    ap.add_argument("--base_url",default="")
    ap.add_argument("--reasoning_effort",default="")
    ap.add_argument("--npz",default=NPZ)
    a=ap.parse_args()

    if a.labelset=="broad":
        CANDS=["T","B","NK","Monocyte"]
    else:
        CANDS=["CD4 T","CD8 T","B","NK","Monocyte"]
    cands_str=", ".join(f"'{c}'" for c in CANDS)+"."

    d=np.load(a.npz, allow_pickle=True)
    G=sparse.csr_matrix((d["data"],d["indices"],d["indptr"]),shape=tuple(d["shape"]))  # cells x genes RNA counts
    genes=d["genes"]
    GA=sparse.csr_matrix((d["ga_data"],d["ga_indices"],d["ga_indptr"]),shape=tuple(d["ga_shape"]))  # cells x ga_genes
    ga_genes=d["ga_genes"]
    labels=d["labels_broad"] if a.labelset=="broad" else d["labels"]

    # TF-IDF on gene-activity (与 ASAP 同, 抑制管家/promoter-dense 偏置)
    N=GA.shape[0]
    df=np.asarray((GA>0).sum(0)).ravel(); idf=np.log((N+1)/(df+1))+1
    rowsum=np.asarray(GA.sum(1)).ravel()+1e-9
    Gc=GA.tocoo(); tf=Gc.data/rowsum[Gc.row]
    TFIDF=sparse.csr_matrix((tf*idf[Gc.col],(Gc.row,Gc.col)),shape=GA.shape)
    GAlog=GA.copy().astype(float); GAlog.data=np.log1p(GAlog.data)  # 用于显示 activity 值
    ga_idx={g:j for j,g in enumerate(ga_genes)}
    panel=[(g,ga_idx[g]) for g in PANEL_GENES if g in ga_idx]

    rng0=np.random.default_rng(a.seed)
    idx_by={}
    for i,l in enumerate(labels):
        if l in CANDS: idx_by.setdefault(l,[]).append(i)
    use=[]
    for l in CANDS:
        ids=idx_by.get(l,[]); rng0.shuffle(ids); use+=ids[:a.n]
    rng0.shuffle(use)
    print(f"model={a.model} cells={len(use)} per-class={ {l:min(len(idx_by.get(l,[])),a.n) for l in CANDS} }",flush=True)

    def rna_sentence(i):
        row=G[i].tocoo(); order=np.argsort(-row.data)[:a.kg]
        return " ".join(genes[row.col[j]] for j in order)
    rna={i:rna_sentence(i) for i in use}

    def atac_block(i):  # top-K gene-activity 基因 (TF-IDF 排序, 显示 log1p activity 值)
        row=TFIDF[i].tocoo()
        if row.nnz==0: return "(none)"
        order=row.col[np.argsort(-row.data)[:a.kp]]
        return " ".join(f"{ga_genes[j]}:{GAlog[i,j]:.1f}" for j in order)
    def panel_block(i):  # 固定 marker panel 的 gene-activity 值 (降序)
        vals=[(g,GAlog[i,j]) for g,j in panel]
        vals.sort(key=lambda x:-x[1])
        return " ".join(f"{g}:{v:.1f}" for g,v in vals)
    atac={i:atac_block(i) for i in use}
    pan={i:panel_block(i) for i in use}

    # 跨细胞置换
    perm=use.copy(); rng0.shuffle(perm); permmap=dict(zip(use,perm))

    arms=a.arms.split(",")
    def build(i, arm):
        head=(f"Below is the data for a single cell measured by 10x Multiome (RNA + ATAC). Select the one cell type it "
              f"most likely belongs to from the following candidate cell types: {cands_str} Make your choice in format "
              f"'Final Answer: Prediction'.")
        genes_part=f"\nGenes (descending expression): {rna[i]}"
        if arm=="l0":          return head+genes_part
        if arm=="l0spec":      return head+"\n"+SPEC_RNA+genes_part
        if arm=="f3atac":      return head+"\n"+SPEC_RNA+"\n"+SPEC_ATAC+genes_part+f"\nChromatin gene-activity (descending): {atac[i]}"
        if arm=="f3atacshuf":  return head+"\n"+SPEC_RNA+"\n"+SPEC_ATAC+genes_part+f"\nChromatin gene-activity (descending): {atac[permmap[i]]}"
        if arm=="f3panel":     return head+"\n"+SPEC_RNA+"\n"+SPEC_PANEL+genes_part+f"\nLineage-gene chromatin accessibility: {pan[i]}"
        if arm=="f3panelshuf": return head+"\n"+SPEC_RNA+"\n"+SPEC_PANEL+genes_part+f"\nLineage-gene chromatin accessibility: {pan[permmap[i]]}"
        raise ValueError(arm)

    def parse_pred(text):
        if not text: return None
        m=re.search(r"Final Answer:\s*(.+)", text, re.I)
        seg=(m.group(1) if m else "").strip().strip("'\"`*. ").lower()
        if seg:
            for c in CANDS:
                if seg==c.lower(): return c
            for c in sorted(CANDS,key=len,reverse=True):
                if c.lower() in seg or seg in c.lower(): return c
        low=text.lower(); hits=[(low.rfind(c.lower()),c) for c in CANDS if c.lower() in low]
        if hits: return sorted(hits,key=lambda h:(h[0],len(h[1])))[-1][1]
        return None

    client,model_name=load_model(a.model, a.base_url)
    tasks=[(i,arm,build(i,arm)) for i in use for arm in arms]
    print(f"total LLM calls={len(tasks)}",flush=True)
    results={}
    def call(t):
        i,arm,u=t
        try:
            kw=dict(model=model_name,messages=[{"role":"system","content":SYS},{"role":"user","content":u}],
                    max_tokens=a.max_tokens, temperature=0)
            if a.reasoning_effort: kw["reasoning_effort"]=a.reasoning_effort
            r=client.chat.completions.create(**kw)
            return (i,arm,parse_pred(r.choices[0].message.content))
        except Exception as e:
            return (i,arm,f"__ERR__{type(e).__name__}")
    t0=time.time(); done=0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call,t) for t in tasks]):
            i,arm,pred=fut.result(); results[(i,arm)]=pred; done+=1
            if done%200==0: print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",flush=True)

    print(f"\n=== accuracy (n={len(use)} cells) ===")
    acc={}
    for arm in arms:
        ok=sum(results.get((i,arm))==labels[i] for i in use)
        err=sum(str(results.get((i,arm))).startswith("__ERR__") for i in use)
        acc[arm]=ok/len(use); print(f"  {arm:12s} acc={ok/len(use)*100:5.1f}%  ({ok}/{len(use)})  err={err}")
    if "l0" in arms:
        base=[results.get((i,"l0"))==labels[i] for i in use]
        for arm in arms:
            if arm=="l0": continue
            oth=[results.get((i,arm))==labels[i] for i in use]
            b01=sum(1 for x,y in zip(base,oth) if (not x) and y); b10=sum(1 for x,y in zip(base,oth) if x and (not y))
            print(f"  l0 vs {arm:12s}: gain(+)={b01} loss(-)={b10} net={b01-b10:+d}")
    # per-class l0 -> f3atac / f3panel
    for fa in ("f3atac","f3panel"):
        if fa in arms and "l0" in arms:
            print(f"\n=== per-class acc l0 -> {fa} ===")
            for c in CANDS:
                ids=[i for i in use if labels[i]==c]
                if not ids: continue
                a0=sum(results.get((i,"l0"))==c for i in ids)/len(ids)
                a3=sum(results.get((i,fa))==c for i in ids)/len(ids)
                print(f"  {c:12s} n={len(ids):3d}  l0={a0*100:5.1f}%  {fa}={a3*100:5.1f}%  Δ={(a3-a0)*100:+5.1f}")
    os.makedirs(a.out,exist_ok=True)
    outp=f"{a.out}/multiome_{a.model}_n{a.n}_{a.labelset}.json"
    # significance.py 格式: arms / truth / preds
    json.dump({"acc":acc,"model":a.model,"n":a.n,"arms":arms,
               "truth":{str(i):labels[i] for i in use},
               "preds":{f"{i}|{arm}":results[(i,arm)] for (i,arm) in results}}, open(outp,"w"))
    print("saved",outp,f"[{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
