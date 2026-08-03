#!/usr/bin/env python3
"""效率分析: L0(纯RNA) vs F3(RNA+蛋白) 同时比较准确率和 token 成本。
假说: F3 输入更长(多了蛋白块), 但模型"身份已显式", 推理(completion) token 更少 → 总成本可能更低,
        且把算力从"脑补身份"挪到"产出答案"。复用 CITE/ASAP 底座与 run_cite_ab 的 prompt 构造。
记录每次调用的 usage(prompt/completion/total tokens) + 是否答对, 聚合出 per-arm 的 acc + token 分布。
用法: .venv/bin/python run_efficiency_ab.py --model gpt54_mini --n 30 \
        [--npz raw_data/mimitou_asap/asap_labeled.npz --modality ASAP-seq]
"""
import os, json, re, time, argparse, yaml
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

NPZ="raw_data/mimitou_cite/cite_labeled.npz"
SYS="You are an expert who knows a lot about single cell biology and genomics and will help me solve a series of tasks related to single cell data analysis."
CANDS=["Naive CD4+ T","Effector CD4+ T","Naive CD8+ T","Effector CD8+ T","B","NK","Monocytes"]
cands_str=", ".join(f"'{c}'" for c in CANDS)+"."
SPEC_RNA=("Note on the gene list: genes are listed in descending order of expression (most expressed first); "
          "only the top expressed genes are shown and a gene's absence does not imply it is biologically off.")
SPEC_PROT=("Note on surface proteins: this is a CITE-seq cell, so besides RNA it was also measured for surface "
           "protein abundance with antibodies. Below the genes, the most abundant surface proteins are listed as "
           "NAME:v (v = CLR-normalized abundance, higher = more protein on the cell surface), in descending order. "
           "Surface proteins are a direct, low-noise readout of cell-surface identity. Use them together with the genes.")

def load_model(name):
    y=yaml.safe_load(open("AtomWorldBench/src/config/models.yaml"))[name]
    key=os.environ.get(y["api_key"].strip("${}"))
    return OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or y["base_url"], timeout=180, max_retries=2), y["model_name"]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="gpt54_mini")
    ap.add_argument("--n",type=int,default=30)
    ap.add_argument("--kg",type=int,default=200)
    ap.add_argument("--kp",type=int,default=30)
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--max_tokens",type=int,default=4096)
    ap.add_argument("--npz",default=NPZ)
    ap.add_argument("--modality",default="CITE-seq")
    ap.add_argument("--out",default="results_efficiency")
    ap.add_argument("--reasoning_effort",default="",help="gpt-5.4-mini: none/low/medium/high/xhigh (默认none不思考); 设high才开CoT")
    a=ap.parse_args()

    d=np.load(a.npz, allow_pickle=True)
    G=sparse.csr_matrix((d["data"],d["indices"],d["indptr"]),shape=tuple(d["shape"]))
    genes=d["genes"]; proteins=d["proteins"]; Pc=d["P_clr"]; labels=d["labels7"]
    rng0=np.random.default_rng(a.seed)
    idx_by={}
    for i,l in enumerate(labels):
        if l in CANDS: idx_by.setdefault(l,[]).append(i)
    use=[]
    for l in CANDS:
        ids=idx_by.get(l,[]); rng0.shuffle(ids); use+=ids[:a.n]
    rng0.shuffle(use)
    print(f"model={a.model} cells={len(use)} modality={a.modality}",flush=True)

    def rna_sentence(i):
        row=G[i].tocoo(); order=np.argsort(-row.data)[:a.kg]
        return " ".join(genes[row.col[j]] for j in order)
    def prot_block(i):
        sub=np.argsort(-Pc[i])[:a.kp]
        return " ".join(f"{proteins[j]}:{Pc[i][j]:.1f}" for j in sub)
    rna={i:rna_sentence(i) for i in use}

    head=(f"Below is the data for a single cell measured by {a.modality}. Select the one cell type it most likely "
          f"belongs to from the following candidate cell types: {cands_str} Make your choice in format "
          f"'Final Answer: Prediction'.")
    def build(i,arm):
        genes_part=f"\nGenes (descending expression): {rna[i]}"
        if arm=="l0": return head+genes_part
        if arm=="f3": return head+"\n"+SPEC_RNA+"\n"+SPEC_PROT+genes_part+f"\nSurface proteins: {prot_block(i)}"
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

    client,model_name=load_model(a.model)
    arms=["l0","f3"]
    tasks=[(i,arm,build(i,arm)) for i in use for arm in arms]
    print(f"total calls={len(tasks)}",flush=True)
    rec={}
    def call(t):
        i,arm,u=t
        try:
            kw=dict(model=model_name,
                messages=[{"role":"system","content":SYS},{"role":"user","content":u}],
                max_tokens=a.max_tokens, temperature=0)
            if a.reasoning_effort: kw["reasoning_effort"]=a.reasoning_effort
            r=client.chat.completions.create(**kw)
            us=r.usage
            comp=getattr(us,"completion_tokens",None); prom=getattr(us,"prompt_tokens",None); tot=getattr(us,"total_tokens",None)
            # 推理模型可能在 completion_tokens_details.reasoning_tokens 暴露思考token
            reas=None
            try: reas=us.completion_tokens_details.reasoning_tokens
            except Exception: pass
            return (i,arm,parse_pred(r.choices[0].message.content),prom,comp,tot,reas)
        except Exception as e:
            return (i,arm,f"__ERR__{type(e).__name__}",None,None,None,None)
    t0=time.time(); done=0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call,t) for t in tasks]):
            i,arm,pred,prom,comp,tot,reas=fut.result()
            rec[(i,arm)]=dict(pred=pred,prompt=prom,completion=comp,total=tot,reasoning=reas)
            done+=1
            if done%100==0: print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]",flush=True)

    print(f"\n=== efficiency (n={len(use)} cells) ===")
    def stat(arm,key):
        vals=[rec[(i,arm)][key] for i in use if rec[(i,arm)].get(key) is not None]
        return (np.mean(vals),np.median(vals)) if vals else (float('nan'),float('nan'))
    summ={}
    for arm in arms:
        ok=sum(rec[(i,arm)]["pred"]==labels[i] for i in use)
        acc=ok/len(use)
        pm,pmd=stat(arm,"prompt"); cm,cmd=stat(arm,"completion"); tm,tmd=stat(arm,"total"); rm,rmd=stat(arm,"reasoning")
        summ[arm]=dict(acc=acc,prompt_mean=pm,completion_mean=cm,total_mean=tm,reasoning_mean=rm,
                       completion_median=cmd,total_median=tmd)
        print(f"  {arm:4s} acc={acc*100:5.1f}%  prompt~{pm:6.0f}  completion~{cm:6.0f}(med {cmd:.0f})  "
              f"reasoning~{rm if rm==rm else float('nan'):6.0f}  total~{tm:6.0f}")
    if "l0" in summ and "f3" in summ:
        print(f"\n  Δcompletion (f3-l0) = {summ['f3']['completion_mean']-summ['l0']['completion_mean']:+.0f} "
              f"({(summ['f3']['completion_mean']/max(summ['l0']['completion_mean'],1e-9)-1)*100:+.1f}%)")
        print(f"  Δtotal      (f3-l0) = {summ['f3']['total_mean']-summ['l0']['total_mean']:+.0f}")
    os.makedirs(a.out,exist_ok=True)
    outp=f"{a.out}/eff_{a.model}_{a.modality.replace('-','')}_n{a.n}.json"
    json.dump({"summary":summ,"model":a.model,"modality":a.modality,
               "rec":{f"{i}|{arm}":rec[(i,arm)] for (i,arm) in rec},
               "labels":{str(i):labels[i] for i in use}}, open(outp,"w"))
    print("saved",outp,f"[{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
