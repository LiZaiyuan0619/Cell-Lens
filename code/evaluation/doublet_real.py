#!/usr/bin/env python3
"""Doublet 检测 A/B 在 **真实公开 doublet** 上验证 (GSE96583, Kang et al. 2018)。
NO simulation: 不相加两个细胞。直接读 demuxlet 基因型真值标注的真实 barcode
  SINGLET = demuxlet multiplets=='singlet' (SNG), DOUBLET = 'doublet' (DBL)。
数据 = pooled PBMC scRNA-seq, RNA-only (无 CITE 蛋白) → 只有 l0 / ccmark / ccshuf (无 ccprot)。
立意/臂/cell-sentence/RNA_MARK 扫描/prompt/parser/McNemar/JSON 全部沿用 doublet_ab.py。
  l0      = 纯 RNA cell-sentence → SINGLET/DOUBLET            [baseline]
  ccmark  = + 显式谱系程序扫描(把被埋的第二程序摊开, flavor②)  [显式化潜结构]
  ccshuf  = ccmark 但程序块取自另一 barcode(跨barcode置换) → 砸穿=证明用真实程序非堆文本 [控制]
真实 doublet 以 HOMOTYPIC(同型)为主, 谱系程序扫描可能抓不到 → 据此解读 (见 caveat)。
用法: set -a; source .env; set +a;
     .venv/bin/python doublet_real.py --model deepseek_v4_flash --n 150 --base_url endpoint3 --workers 16
     .venv/bin/python doublet_real.py --dry   # 只打印构造样例(不调API)
"""
import os, json, re, time, argparse, yaml
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

NPZ = "raw_data/gse96583/doublet_real_labeled.npz"
SYS = ("You are an expert who knows a lot about single cell biology and genomics and will help me "
       "solve a series of tasks related to single cell data analysis.")

# 4 大谱系的 RNA 经典标记 (沿用 doublet_ab.py 的 RNA_MARK)
RNA_MARK = {"T":["CD3D","CD3E","CD3G","TRAC","TRBC2","CD8A","IL7R"],
            "B":["MS4A1","CD79A","CD79B","CD19","BANK1"],
            "NK":["NKG7","GNLY","KLRD1","NCAM1","KLRF1"],
            "Mono":["CD14","LYZ","FCGR3A","S100A8","S100A9","CST3"]}
LINS = ["T","B","NK","Mono"]

EPS = {"endpoint1":"https://api.example.com/v1","endpoint2":"https://api.example.com/v1",
       "endpoint3":"https://api.example.com/v1"}
def load_model(name, base_url=""):
    y = yaml.safe_load(open("AtomWorldBench/src/config/models.yaml"))[name]
    key = os.environ.get(y["api_key"].strip("${}"))
    url = EPS.get(base_url, base_url) or os.environ.get("LLM_BASE_URL") or y["base_url"]
    return OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or url, timeout=180, max_retries=2), y["model_name"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek_v4_flash")
    ap.add_argument("--n", type=int, default=150, help="参与的总 barcode 数上限 (npz 已平衡 150+150)")
    ap.add_argument("--kg", type=int, default=200, help="RNA top基因数")
    ap.add_argument("--arms", default="l0,ccmark,ccshuf")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--out", default="results_doublet_real")
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--modality", default="scRNA-seq")
    ap.add_argument("--base_url", default="endpoint3")
    ap.add_argument("--reasoning_effort", default="")
    ap.add_argument("--min_mark", type=int, default=2, help="一个谱系程序判'present'所需的已表达标记数")
    ap.add_argument("--dry", action="store_true", help="只打印构造样例, 不调API")
    a = ap.parse_args()

    d = np.load(a.npz, allow_pickle=True)
    G = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])).tocsr()
    genes = list(d["genes"]); labels = list(d["labels"])          # 真实标签: 'SINGLET'/'DOUBLET'
    celltype = list(d["celltype"]) if "celltype" in d.files else ["?"]*len(labels)
    barcodes = list(d["barcodes"]) if "barcodes" in d.files else [str(i) for i in range(len(labels))]
    gset = set(genes); gi = {g: k for k, g in enumerate(genes)}
    rng = np.random.default_rng(a.seed)

    # 平衡子采样到 n+n (若 npz 已是 150+150 且 --n>=150 则全用)
    idx_s = [i for i, l in enumerate(labels) if l == "SINGLET"]
    idx_d = [i for i, l in enumerate(labels) if l == "DOUBLET"]
    take = min(a.n, len(idx_s), len(idx_d))
    sel = (list(rng.choice(idx_s, size=take, replace=False)) +
           list(rng.choice(idx_d, size=take, replace=False)))
    order = list(rng.permutation(len(sel))); cells = [int(sel[i]) for i in order]
    truth = [labels[c] for c in cells]
    print(f"model={a.model} REAL GSE96583  barcodes={len(cells)} "
          f"(singlet={truth.count('SINGLET')} doublet={truth.count('DOUBLET')})", flush=True)

    # —— 真实 barcode: 直接读其自身 RNA 计数 (无相加) ——
    def counts(k):
        return G[cells[k]]
    def rna_sentence(k):
        row = counts(k).tocoo(); ordr = np.argsort(-row.data)[:a.kg]
        return " ".join(genes[row.col[j]] for j in ordr)
    def programs_present(k):
        # 从全表达(非仅top-K)判定: 一个谱系若 >=min_mark 个标记被检出(count>0)→present
        row = counts(k).tocoo(); expr = set(int(c) for c, v in zip(row.col, row.data) if v > 0)
        out = []
        for lin in LINS:
            hit = [m for m in RNA_MARK[lin] if m in gset and gi[m] in expr]
            if len(hit) >= a.min_mark: out.append((lin, hit))
        return out

    HEAD = (f"Below is the data for ONE droplet from a single-cell {a.modality} experiment. A droplet usually "
            f"contains one cell (a SINGLET), but sometimes two cells are captured together (a DOUBLET / multiplet). "
            f"Decide whether this droplet is a SINGLET or a DOUBLET. "
            f"Answer in the format 'Final Answer: SINGLET' or 'Final Answer: DOUBLET'.")
    SPEC_RNA = ("Note on the gene list: genes are listed in descending order of expression; only top genes are shown, "
                "so a second cell's markers can be pushed below the cutoff and become hard to see.")
    MARK_NOTE = ("Note: a single cell expresses ONE lineage program. The scan below reports which canonical lineage "
                 "programs are detectable in this droplet (including markers expressed but ranked low). If TWO "
                 "mutually-exclusive lineage programs are both present, the droplet most likely contains two cells.")
    permmap = list(range(len(cells))); rng.shuffle(permmap)  # ccshuf 用

    def mark_text(k):
        progs = programs_present(k)
        if not progs: return "(no clear lineage program detected)"
        return "; ".join(f"{lin}-cell program: {', '.join(hit)} [present]" for lin, hit in progs)
    def build(k, arm):
        rs = rna_sentence(k)
        genes_part = f"\nGenes (descending expression): {rs}"
        if arm == "l0":
            return HEAD + genes_part
        if arm == "ccmark":
            return (HEAD + "\n" + SPEC_RNA + "\n" + MARK_NOTE + genes_part +
                    f"\nLineage-program scan: {mark_text(k)}")
        if arm == "ccshuf":
            return (HEAD + "\n" + SPEC_RNA + "\n" + MARK_NOTE + genes_part +
                    f"\nLineage-program scan: {mark_text(permmap[k])}")
        raise ValueError(arm)

    def parse_pred(text):
        if not text: return None
        m = re.search(r"Final Answer:\s*(.+)", text, re.I)
        seg = (m.group(1) if m else text).strip().upper()
        hits = [(seg.rfind("DOUBLET"), "DOUBLET"), (seg.rfind("SINGLET"), "SINGLET")]
        hits = [(p, w) for p, w in hits if p >= 0]
        if hits: return sorted(hits)[-1][1]
        low = text.upper(); h2 = [(low.rfind("DOUBLET"), "DOUBLET"), (low.rfind("SINGLET"), "SINGLET")]
        h2 = [(p, w) for p, w in h2 if p >= 0]
        return sorted(h2)[-1][1] if h2 else None

    arms = a.arms.split(",")
    if a.dry:
        print("\n===== 构造样例(各 2 singlet + 2 doublet, 真实 demuxlet 标签) =====")
        ex = [k for k in range(len(cells)) if truth[k]=="SINGLET"][:2] + \
             [k for k in range(len(cells)) if truth[k]=="DOUBLET"][:2]
        for k in ex:
            print(f"\n--- barcode {barcodes[cells[k]]}  truth={truth[k]}  (demuxlet primary-type={celltype[cells[k]]}) ---")
            print(f"  programs_present: {[(l,h) for l,h in programs_present(k)]}")
            print(f"  [ccmark scan]: {mark_text(k)}")
        import collections
        for tv in ["SINGLET","DOUBLET"]:
            cnts = collections.Counter(len(programs_present(k)) for k in range(len(cells)) if truth[k]==tv)
            print(f"\n{tv}: #检出谱系程序数分布 = {dict(sorted(cnts.items()))}")
        return

    client, model_name = load_model(a.model, a.base_url)
    tasks = [(k, arm, build(k, arm)) for k in range(len(cells)) for arm in arms]
    print(f"total LLM calls={len(tasks)}", flush=True)
    results = {}
    def call(t):
        k, arm, u = t
        try:
            kw = dict(model=model_name, messages=[{"role":"system","content":SYS},{"role":"user","content":u}],
                      max_tokens=a.max_tokens, temperature=0)
            if a.reasoning_effort: kw["reasoning_effort"] = a.reasoning_effort
            r = client.chat.completions.create(**kw)
            return (k, arm, parse_pred(r.choices[0].message.content))
        except Exception as e:
            return (k, arm, f"__ERR__{type(e).__name__}")
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call, t) for t in tasks]):
            k, arm, pred = fut.result(); results[(k, arm)] = pred; done += 1
            if done % 100 == 0: print(f"  {done}/{len(tasks)} [{time.time()-t0:.0f}s]", flush=True)

    nS = truth.count("SINGLET"); nD = truth.count("DOUBLET")
    print(f"\n=== REAL doublet 检测 GSE96583 (n={len(cells)}: {nS} singlet + {nD} doublet) ===")
    acc = {}
    for arm in arms:
        ok = sum(results.get((k, arm)) == truth[k] for k in range(len(cells)))
        err = sum(str(results.get((k, arm))).startswith("__ERR__") for k in range(len(cells)))
        sok = sum(results.get((k, arm)) == "SINGLET" for k in range(len(cells)) if truth[k] == "SINGLET")
        dok = sum(results.get((k, arm)) == "DOUBLET" for k in range(len(cells)) if truth[k] == "DOUBLET")
        acc[arm] = ok / len(cells)
        print(f"  {arm:8s} acc={ok/len(cells)*100:5.1f}%  singlet-recall={sok/nS*100:5.1f}%  "
              f"doublet-recall={dok/nD*100:5.1f}%  err={err}")
    if "l0" in arms:
        base = [results.get((k, "l0")) == truth[k] for k in range(len(cells))]
        for arm in arms:
            if arm == "l0": continue
            oth = [results.get((k, arm)) == truth[k] for k in range(len(cells))]
            b01 = sum(1 for x, y in zip(base, oth) if (not x) and y)
            b10 = sum(1 for x, y in zip(base, oth) if x and (not y))
            print(f"  l0 vs {arm}: gain(+)={b01} loss(-)={b10} net={b01-b10:+d}")
    os.makedirs(a.out, exist_ok=True)
    outp = f"{a.out}/doublet_real_{a.model}_n{len(cells)//2}.json"
    json.dump({"acc": acc, "model": a.model, "n": len(cells)//2, "arms": arms, "dataset": "GSE96583",
               "preds": {f"{k}|{arm}": results[(k, arm)] for (k, arm) in results},
               "truth": {str(k): truth[k] for k in range(len(cells))}}, open(outp, "w"))
    print("saved", outp, f"[{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
