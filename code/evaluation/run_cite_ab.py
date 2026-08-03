#!/usr/bin/env python3
"""F3/CITE 表示 A/B harness。在 Mimitou CITE 配对底座上, 同一批细胞构造多臂表示, 并发调 LLM, exact-match + McNemar。
核心命题: CellVerse 把 CITE 的蛋白模态在输入端压平成 RNA 基因名丢掉了; 我们把蛋白恢复回来(F3)。
臂:
 l0      = 纯 RNA cell-sentence(top-K 基因, 降序) [复现 CellVerse CITE 基线]
 l0spec  = l0 + RNA 说明(控制"多了说明书")
 f3      = l0 + 本细胞蛋白块(top-Kp 蛋白 by CLR, NAME:value 降序) + 蛋白说明 [主臂: 恢复蛋白模态]
 f3shuf  = l0 + 随机另一细胞的蛋白块(跨细胞置换, 控制"蛋白对应关系是否真起作用")
 f3noval = l0 + 蛋白名(CLR 降序, 无数值) [测是数值还是仅排序/身份在起作用]
用法: .venv/bin/python run_cite_ab.py --model deepseek_v4_flash --n 60 --kg 200 --kp 30 --arms l0,l0spec,f3,f3shuf
"""
import os, json, re, time, argparse, random, yaml
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
SPEC_NEG=("Note on negative evidence: some canonical lineage-defining surface proteins are measured but found "
          "LOW/absent on this cell. Their absence is informative for ruling out lineages (e.g., no CD3 rules out T cells).")
SPEC_FILLER=("General background on single-cell measurements: single-cell sequencing profiles individual cells and the "
             "resulting data are often sparse and noisy due to limited molecular capture; cell identities lie along "
             "continuous developmental and activation states rather than fully discrete categories; expression is shaped "
             "by transcriptional and post-transcriptional regulation as well as the microenvironment; technical factors "
             "such as sequencing depth, dropout, and batch effects can affect the measured profile; weighing the overall "
             "evidence carefully tends to give more reliable cell-type assignments.")
SPEC_LINKS=("Cross-modal identity links: the SAME lineage identity is measured BOTH as a surface protein AND as gene(s); "
            "use the protein and its gene(s) together as corroborating evidence. "
            "CD3 protein <-> CD3D/CD3E/CD3G genes (T cells); CD4 protein <-> CD4 gene (helper T); "
            "CD8 protein <-> CD8A/CD8B genes (cytotoxic T); CD19/CD20 protein <-> CD19/MS4A1 genes (B cells); "
            "CD14 protein <-> CD14/LYZ genes (monocytes); CD56 protein <-> NCAM1/NKG7 genes (NK); "
            "CD45RA protein <-> naive state, CD45RO protein <-> memory/effector state.")

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
    ap.add_argument("--n",type=int,default=60,help="每类最多取多少细胞")
    ap.add_argument("--kg",type=int,default=200,help="RNA top基因数")
    ap.add_argument("--kp",type=int,default=30,help="蛋白 top数")
    ap.add_argument("--arms",default="l0,l0spec,f3,f3shuf")
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--max_tokens",type=int,default=2048)
    ap.add_argument("--out",default="results_cite")
    ap.add_argument("--labelset",default="7",choices=["7","broad","9","bmmc","tall","heart"],
                    help="7=CellVerse匹配7类(蛋白门控); broad=RNA派生T/B/NK/Mono(与蛋白独立,循环性对冲); "
                         "9=cite_aligned.npz 9类含rare(DC/Mono/NK), rarity梯度分层; "
                         "bmmc=GSE194122骨髓11粗类(含PBMC没有的Erythroid/HSPC=非PBMC广度, 传 --npz bmmc_labeled.npz)")
    ap.add_argument("--npz",default=NPZ,help="底座npz(默认CITE; ASAP传asap_labeled.npz)")
    ap.add_argument("--modality",default="CITE-seq",help="prompt里写的模态名(CITE-seq / ASAP-seq)")
    ap.add_argument("--native_fmt",action="store_true",help="用原生CellVerse措辞(忠实度对齐, 修gemini偏低)")
    ap.add_argument("--base_url",default="",help="API endpoint alias or URL; empty uses the YAML default")
    ap.add_argument("--reasoning_effort",default="",help="gpt-5.4-mini: low/medium/high/xhigh(默认none不思考); 测rneg是否需推理")
    a=ap.parse_args()
    global CANDS, cands_str
    if a.labelset=="broad":
        CANDS=["T","B","NK","Monocyte"]; cands_str=", ".join(f"'{c}'" for c in CANDS)+"."
    elif a.labelset=="9":
        # cite_aligned.npz 'labels' 字段的 9 类 (含真实rarity梯度: B=3756 ... DC=12)
        CANDS=["B","Naive CD4 T","Effector/Memory CD4 T","Naive CD8 T","Effector/Memory CD8 T",
               "NK","Monocyte","DC"]  # 排除 Unassigned (非真类型)
        cands_str=", ".join(f"'{c}'" for c in CANDS)+"."
    elif a.labelset=="bmmc":
        # GSE194122 骨髓 11 粗类 (build_bmmc.py 映射), 含 PBMC 没有的 Erythroid/HSPC/Prog = 非PBMC广度
        CANDS=["CD14+ Mono","CD16+ Mono","CD4 T","CD8 T","NK","B","Plasma","Erythroid","HSPC/Prog","cDC","pDC"]
        cands_str=", ".join(f"'{c}'" for c in CANDS)+"."
    elif a.labelset=="tall":
        # GSE248287 T-ALL 7 谱系 (Celltypes_initial→自然名), 疾病组织=非PBMC广度。
        # Malignant(母细胞)剔除: 是疾病状态非谱系, 模型永远判成 T/Myeloid → 不公平 class。
        CANDS=["T cell","B cell","NK cell","Monocyte","Dendritic cell",
               "Plasmacytoid dendritic cell","Hematopoietic stem cell"]
        cands_str=", ".join(f"'{c}'" for c in CANDS)+"."
    elif a.labelset=="heart":
        # GSE217494 心脏 7 类 (实体器官=最强广度, 含血/髓没有的结构细胞 Endo/Fibro/SMC/Endocardium)
        CANDS=["Endothelial cell","Fibroblast","Myeloid cell","Smooth muscle cell",
               "T cell","Endocardial cell","B cell"]
        cands_str=", ".join(f"'{c}'" for c in CANDS)+"."

    d=np.load(a.npz, allow_pickle=True)
    G=sparse.csr_matrix((d["data"],d["indices"],d["indptr"]),shape=tuple(d["shape"]))  # cells x genes (raw counts)
    genes=d["genes"]; proteins=d["proteins"]; Pc=d["P_clr"]
    labels=(d["labels_rna"] if a.labelset=="broad" else
            d["labels"] if a.labelset=="9" else
            d["labels_init"] if a.labelset in ("tall","heart") else d["labels7"])
    rng0=np.random.default_rng(a.seed)
    # 每类均衡抽样(排除 Unassigned)
    idx_by={}
    for i,l in enumerate(labels):
        if l in CANDS: idx_by.setdefault(l,[]).append(i)
    use=[]
    for l in CANDS:
        ids=idx_by.get(l,[]); rng0.shuffle(ids); use+=ids[:a.n]
    rng0.shuffle(use)
    print(f"model={a.model} cells={len(use)} per-class={ {l:min(len(idx_by.get(l,[])),a.n) for l in CANDS} }",flush=True)

    # 预构造每个细胞的 RNA sentence + 蛋白块
    def rna_sentence(i):
        row=G[i].tocoo(); order=np.argsort(-row.data)[:a.kg]
        return " ".join(genes[row.col[j]] for j in order)
    # 定义 labels7 的门控蛋白(f3ablate 剔除这些, 测非循环信号)
    GATING=set(["CD3.1","CD3.2","CD4.1","CD4.2","CD8","CD8a","CD19","CD20","CD14",
                "CD56.NCAM.","CD56.NCAM.Recombinant","CD335","CD45RA","CD45RO"])
    prot_keep=np.array([j for j in range(len(proteins)) if proteins[j] not in GATING])
    def prot_block(i, withval=True, exclude_gating=False):
        pool=prot_keep if exclude_gating else np.arange(len(proteins))
        sub=pool[np.argsort(-Pc[i][pool])[:a.kp]]
        if withval:
            return " ".join(f"{proteins[j]}:{Pc[i][j]:.1f}" for j in sub)
        return " ".join(f"{proteins[j]}" for j in sub)
    # F2 阴性证据: 固定谱系 panel 里, 该细胞 CLR 低于群体均值的 = "明确低/缺"
    LINEAGE=["CD3.1","CD4.1","CD8a","CD19","CD20","CD14","CD56.NCAM.","CD335","CD11c","CD123","CD45RA","CD45RO"]
    lin_idx=[(nm,proteins.tolist().index(nm)) for nm in LINEAGE if nm in proteins.tolist()]
    Pmean=Pc.mean(0); Pstd=Pc.std(0)+1e-9
    def neg_block(i):  # 只列"明确缺失"(z<-1, 真低于该蛋白群体分布), 避免把相对低的defining marker误标
        low=[nm for nm,j in lin_idx if (Pc[i][j]-Pmean[j])/Pstd[j] < -1.0]
        return ", ".join(low) if low else "(none)"
    rna={i:rna_sentence(i) for i in use}
    # ── 单模态探针(扩 CellCode 出多模态): 恢复 cell-sentence(rank-only top-K) 丢弃的 ①幅度 ②零/缺失 ──
    def rna_sentence_val(i):  # rf1: 每基因附 log1p(UMI) 幅度
        row=G[i].tocoo(); order=np.argsort(-row.data)[:a.kg]
        return " ".join(f"{genes[row.col[j]]}:{np.log1p(row.data[j]):.1f}" for j in order)
    RNA_LINEAGE=["CD3D","CD3E","CD3G","CD4","CD8A","CD8B","MS4A1","CD79A","CD79B","CD19","NKG7","GNLY",
                 "KLRD1","NCAM1","CD14","LYZ","FCGR3A","CCR7","SELL","GZMB","GZMK","IL7R","TCF7","CCL5"]
    gidx={g:k for k,g in enumerate(list(genes))}
    rna_lin=[(g,gidx[g]) for g in RNA_LINEAGE if g in gidx]
    Gcsr=G.tocsr()
    def rna_neg(i):  # rneg: 固定谱系基因中该细胞 0 计数的 = "明确未检出"(top-K rank 丢弃了这个判别信息)
        absent=[g for g,j in rna_lin if Gcsr[i,j]==0]
        return ", ".join(absent) if absent else "(none)"
    # ── 数据驱动 panel 控制(rnegdd): panel 不再人工挑, 而从 train split(排除测试细胞 use, 防泄漏)
    #    按类特异性(class_mean - 次高class_mean)自动选谱系 marker → 杀"你注入了 marker 知识"= rneg版f3ablate ──
    def build_dd_panel(per_class=3, min_mean=0.1):
        use_set=set(use)
        train_by={c:[] for c in CANDS}
        for i in range(G.shape[0]):
            if labels[i] in CANDS and i not in use_set: train_by[labels[i]].append(i)
        cmean=[(np.asarray(G[train_by[c]].mean(0)).ravel() if train_by[c] else np.zeros(G.shape[1])) for c in CANDS]
        M=np.vstack(cmean)  # classes x genes (train-only mean counts)
        panel=set()
        for ci in range(len(CANDS)):
            others=np.delete(M,ci,axis=0).max(0)
            spec=M[ci]-others  # 该类相对其它类最高的特异性(数据驱动marker)
            cand=[int(j) for j in np.argsort(-spec) if M[ci][j]>min_mean][:per_class]
            panel.update(cand)
        return [(genes[j], j) for j in sorted(panel)]
    rna_lin_dd=build_dd_panel() if "rnegdd" in a.arms else []
    if rna_lin_dd: print(f"data-driven panel ({len(rna_lin_dd)} genes, leakage-safe): {[g for g,_ in rna_lin_dd]}",flush=True)
    def rna_neg_dd(i):
        absent=[g for g,j in rna_lin_dd if Gcsr[i,j]==0]
        return ", ".join(absent) if absent else "(none)"
    # ── 改进版数据驱动panel控制(rnegdd2): 按检出频率选"缺失即判别"marker(类内检出高+他类检出低),
    #    自动剔除管家基因(到处检出→他类频率高→排除) = 更公允的"非注入知识"控制 ──
    def build_dd_panel2(per_class=4, min_frac=0.4, max_other=0.25):
        use_set=set(use)
        train_by={c:[] for c in CANDS}
        for i in range(G.shape[0]):
            if labels[i] in CANDS and i not in use_set: train_by[labels[i]].append(i)
        binG=(Gcsr>0)
        F=np.vstack([(np.asarray(binG[train_by[c]].mean(0)).ravel() if train_by[c] else np.zeros(G.shape[1])) for c in CANDS])
        panel=set()
        for ci in range(len(CANDS)):
            others=np.delete(F,ci,axis=0).max(0)  # 任一他类的最高检出频率
            cand=[int(j) for j in np.argsort(-(F[ci]-others)) if F[ci][j]>min_frac and others[j]<max_other][:per_class]
            panel.update(cand)
        return [(genes[j], j) for j in sorted(panel)]
    rna_lin_dd2=build_dd_panel2() if "rnegdd2" in a.arms else []
    if rna_lin_dd2: print(f"data-driven panel2 ({len(rna_lin_dd2)} genes, absence-informative): {[g for g,_ in rna_lin_dd2]}",flush=True)
    def rna_neg_dd2(i):
        absent=[g for g,j in rna_lin_dd2 if Gcsr[i,j]==0]
        return ", ".join(absent) if absent else "(none)"
    # ── 标准 one-vs-rest panel(rnegdd3): 单细胞领域标准做法(每类 vs 全部其它类合并, top-N up-marker, 取并集),
    #    能选回横跨亚型的共享判别marker(CD8A/CD3D/CCR7/SELL/TCF7) = 最公允的"非手挑"自动化控制(scanpy同款思路) ──
    def build_dd_panel3(per_class=5, min_frac=0.25):
        use_set=set(use)
        train_by={c:[] for c in CANDS}
        for i in range(G.shape[0]):
            if labels[i] in CANDS and i not in use_set: train_by[labels[i]].append(i)
        Glog=Gcsr.copy().astype(float); Glog.data=np.log1p(Glog.data)
        cmean={c:(np.asarray(Glog[train_by[c]].mean(0)).ravel() if train_by[c] else np.zeros(G.shape[1])) for c in CANDS}
        panel=set()
        for c in CANDS:
            rest=[i for cc in CANDS if cc!=c for i in train_by[cc]]
            rmean=np.asarray(Glog[rest].mean(0)).ravel()
            fc=np.asarray((Gcsr[train_by[c]]>0).mean(0)).ravel()
            score=cmean[c]-rmean  # one-vs-(pooled rest) log-fold
            cand=[int(j) for j in np.argsort(-score) if fc[j]>min_frac][:per_class]
            panel.update(cand)
        return [(genes[j], j) for j in sorted(panel)]
    rna_lin_dd3=build_dd_panel3() if "rnegdd3" in a.arms else []
    if rna_lin_dd3: print(f"data-driven panel3 ({len(rna_lin_dd3)} genes, one-vs-rest std): {[g for g,_ in rna_lin_dd3]}",flush=True)
    def rna_neg_dd3(i):
        absent=[g for g,j in rna_lin_dd3 if Gcsr[i,j]==0]
        return ", ".join(absent) if absent else "(none)"
    rna_val={i:rna_sentence_val(i) for i in use}
    # 跨细胞置换映射(f3shuf: 细胞 i 用 perm[i] 的蛋白块)
    perm=use.copy(); rng0.shuffle(perm); permmap=dict(zip(use,perm))

    arms=a.arms.split(",")
    def build(i, arm):
        if a.native_fmt:
            head=(f"Below is a list of {{gene_names}} ordered by descending expression level in a single-cell "
                  f"{a.modality} sample. Your task is to select one cell type to which this cell most likely belongs "
                  f"from the following candidate cell types: {cands_str} Make your choice in format 'Final Answer: Prediction'.")
            genes_part=f"\n{{gene_names}}: {rna[i]}"
        else:
            head=(f"Below is the data for a single cell measured by {a.modality}. Select the one cell type it most likely "
                  f"belongs to from the following candidate cell types: {cands_str} Make your choice in format "
                  f"'Final Answer: Prediction'.")
            genes_part=f"\nGenes (descending expression): {rna[i]}"
        if arm=="l0":      return head+genes_part
        if arm=="l0spec":  return head+"\n"+SPEC_RNA+genes_part
        if arm=="rf1":     # 单模态: 恢复幅度
            vp=(f"\n{{gene_names}} (each as NAME:v, v=log(1+UMI count)): {rna_val[i]}" if a.native_fmt
                else f"\nGenes (descending expression, NAME:v where v=log(1+UMI count)): {rna_val[i]}")
            return head+"\n"+SPEC_RNA+vp
        if arm=="rneg":    # 单模态: 恢复被丢弃的零/缺失
            return (head+"\n"+SPEC_RNA+"\nNote: lineage-defining marker genes that were MEASURED but had zero counts "
                    "(not detected) in this cell are listed below the gene list; their absence helps rule out lineages."
                    +genes_part+f"\nLineage marker genes measured but NOT detected (zero counts): {rna_neg(i)}")
        if arm=="rnegshuf":  # 对照: 用别的细胞的"未检出"列表(跨细胞置换) → 砸穿=证明rneg用本细胞真零非多token
            return (head+"\n"+SPEC_RNA+"\nNote: lineage-defining marker genes that were MEASURED but had zero counts "
                    "(not detected) in this cell are listed below the gene list; their absence helps rule out lineages."
                    +genes_part+f"\nLineage marker genes measured but NOT detected (zero counts): {rna_neg(permmap[i])}")
        if arm=="rnegdd":  # 对照: 数据驱动 panel(非人工挑, train-split选)的零 → 仍涨=证明不靠注入marker知识
            return (head+"\n"+SPEC_RNA+"\nNote: lineage-defining marker genes that were MEASURED but had zero counts "
                    "(not detected) in this cell are listed below the gene list; their absence helps rule out lineages."
                    +genes_part+f"\nLineage marker genes measured but NOT detected (zero counts): {rna_neg_dd(i)}")
        if arm=="rnegdd2":  # 对照(改进): absence-informative 数据驱动panel(剔管家) → 更公允的"非注入知识"控制
            return (head+"\n"+SPEC_RNA+"\nNote: lineage-defining marker genes that were MEASURED but had zero counts "
                    "(not detected) in this cell are listed below the gene list; their absence helps rule out lineages."
                    +genes_part+f"\nLineage marker genes measured but NOT detected (zero counts): {rna_neg_dd2(i)}")
        if arm=="rnegdd3":  # 对照(标准): one-vs-rest 数据驱动panel(领域标准, 含共享亚型marker) = 最公允"非手挑"控制
            return (head+"\n"+SPEC_RNA+"\nNote: lineage-defining marker genes that were MEASURED but had zero counts "
                    "(not detected) in this cell are listed below the gene list; their absence helps rule out lineages."
                    +genes_part+f"\nLineage marker genes measured but NOT detected (zero counts): {rna_neg_dd3(i)}")
        if arm=="f3":      return head+"\n"+SPEC_RNA+"\n"+SPEC_PROT+genes_part+f"\nSurface proteins: {prot_block(i)}"
        if arm=="f3noval": return head+"\n"+SPEC_RNA+"\n"+SPEC_PROT+genes_part+f"\nSurface proteins: {prot_block(i,False)}"
        if arm=="f3shuf":  return head+"\n"+SPEC_RNA+"\n"+SPEC_PROT+genes_part+f"\nSurface proteins: {prot_block(permmap[i])}"
        if arm=="f3ablate":return head+"\n"+SPEC_RNA+"\n"+SPEC_PROT+genes_part+f"\nSurface proteins: {prot_block(i,exclude_gating=True)}"
        if arm=="f3links": # 新意承重墙: f3 + 显式跨模态LINKS(同一身份蛋白⟷基因) → 测显式结构是否>简单拼接
            return head+"\n"+SPEC_RNA+"\n"+SPEC_PROT+"\n"+SPEC_LINKS+genes_part+f"\nSurface proteins: {prot_block(i)}"
        if arm=="f3filler": # verbosity对照: f3 + 等长但无跨模态结构的通用文本 → f3links>f3filler 证明是结构非"多了字"
            return head+"\n"+SPEC_RNA+"\n"+SPEC_PROT+"\n"+SPEC_FILLER+genes_part+f"\nSurface proteins: {prot_block(i)}"
        if arm=="f2neg":   return head+"\n"+SPEC_RNA+"\n"+SPEC_PROT+"\n"+SPEC_NEG+genes_part+f"\nSurface proteins: {prot_block(i)}\nLineage proteins notably LOW/absent on this cell: {neg_block(i)}"
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
            kw=dict(model=model_name,
                messages=[{"role":"system","content":SYS},{"role":"user","content":u}],
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
        acc[arm]=ok/len(use); print(f"  {arm:8s} acc={ok/len(use)*100:5.1f}%  ({ok}/{len(use)})  err={err}")
    base_arm="l0spec" if "l0spec" in arms else "l0"
    if base_arm in arms:
        base=[results.get((i,base_arm))==labels[i] for i in use]
        for arm in arms:
            if arm==base_arm: continue
            oth=[results.get((i,arm))==labels[i] for i in use]
            b01=sum(1 for x,y in zip(base,oth) if (not x) and y); b10=sum(1 for x,y in zip(base,oth) if x and (not y))
            print(f"  {base_arm} vs {arm}: gain(+)={b01} loss(-)={b10} net={b01-b10:+d}")
    # 按类拆 f3 vs l0
    if "f3" in arms and "l0" in arms:
        print("\n=== per-class acc l0 -> f3 ===")
        import collections
        ct=collections.Counter(labels[i] for i in use)
        for c in CANDS:
            ids=[i for i in use if labels[i]==c]
            if not ids: continue
            a0=sum(results.get((i,"l0"))==c for i in ids)/len(ids)
            a3=sum(results.get((i,"f3"))==c for i in ids)/len(ids)
            print(f"  {c:18s} n={len(ids):3d}  l0={a0*100:5.1f}%  f3={a3*100:5.1f}%  Δ={(a3-a0)*100:+5.1f}")
    os.makedirs(a.out,exist_ok=True)
    outp=f"{a.out}/cite_{a.model}_n{a.n}_{a.labelset}.json"
    json.dump({"acc":acc,"model":a.model,"n":a.n,"arms":arms,
               "preds":{f"{i}|{arm}":results[(i,arm)] for (i,arm) in results},
               "labels":{str(i):labels[i] for i in use}}, open(outp,"w"))
    print("saved",outp,f"[{time.time()-t0:.0f}s]")

if __name__=="__main__": main()
