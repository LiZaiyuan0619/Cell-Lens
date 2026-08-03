#!/usr/bin/env python3
"""Doublet 解释质量(应用闭环·可解释 QC 审计器, T1.4)。
不仅判 SINGLET/DOUBLET, 还让 LLM 说出它认为存在的谱系。score:
  - detection acc(同 doublet_ab)
  - explanation acc = 在"真doublet且被正确判为DOUBLET"中, 模型是否正确点出**两个真谱系**。
立意: CellCode 不只让冻结 LLM 判对, 还让它**给出可核对的理由**(同时表达 T 和 B 程序)→ 答"谁用冻结LLM读基因表"=可解释 QC 审计器。
臂: l0 / ccmark。用法: .venv/bin/python doublet_explain.py --model deepseek_v4_flash --n 120 --base_url endpoint3
"""
import os, json, re, argparse, yaml
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

NPZ = "raw_data/mimitou_cite/cite_labeled.npz"
SYS = "You are an expert in single cell biology and genomics."
LIN_OF = {"Naive CD4+ T":"T","Effector CD4+ T":"T","Naive CD8+ T":"T","Effector CD8+ T":"T","B":"B","NK":"NK","Monocytes":"Mono"}
RNA_MARK = {"T":["CD3D","CD3E","CD3G","TRAC","TRBC2","CD8A","IL7R"],"B":["MS4A1","CD79A","CD79B","CD19","BANK1"],
            "NK":["NKG7","GNLY","KLRD1","NCAM1","KLRF1"],"Mono":["CD14","LYZ","FCGR3A","S100A8","S100A9","CST3"]}
LINS_ALL = ["T","B","NK","Mono"]
EPS = {"endpoint1":"https://api.example.com/v1","endpoint2":"https://api.example.com/v1","endpoint3":"https://api.example.com/v1"}
def load_model(name, base_url=""):
    y = yaml.safe_load(open("AtomWorldBench/src/config/models.yaml"))[name]
    return OpenAI(api_key=os.environ.get(y["api_key"].strip("${}")), base_url=os.environ.get("LLM_BASE_URL") or EPS.get(base_url, base_url) or y["base_url"], timeout=180, max_retries=2), y["model_name"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek_v4_flash"); ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--kg", type=int, default=200); ap.add_argument("--arms", default="l0,ccmark")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=8192); ap.add_argument("--base_url", default="endpoint3")
    ap.add_argument("--reasoning_effort", default=""); ap.add_argument("--min_mark", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0, help="Claude thinking 需 temp=1")
    ap.add_argument("--out", default="results_doublet_explain")
    a = ap.parse_args()
    d = np.load(NPZ, allow_pickle=True)
    G = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])).tocsr()
    genes = list(d["genes"]); labels = d["labels7"]; gi = {g: k for k, g in enumerate(genes)}; gset = set(genes)
    rng = np.random.default_rng(a.seed)
    by = {}
    for i, l in enumerate(labels):
        lin = LIN_OF.get(l)
        if lin: by.setdefault(lin, []).append(i)
    lins = [k for k in LINS_ALL if by.get(k)]
    pool = [i for l in lins for i in by[l]]
    sidx = rng.choice(pool, size=a.n, replace=False)
    bars = [("S", int(i), None) for i in sidx]
    for _ in range(a.n):
        l1, l2 = rng.choice(lins, size=2, replace=False)
        bars.append(("D", int(rng.choice(by[l1])), int(rng.choice(by[l2]))))
    rng.shuffle(bars)
    truth = ["DOUBLET" if b[0] == "D" else "SINGLET" for b in bars]
    true_lins = [tuple(sorted({LIN_OF[labels[b[1]]], LIN_OF[labels[b[2]]]})) if b[0] == "D" else (LIN_OF[labels[b[1]]],) for b in bars]
    def counts(b): return G[b[1]] if b[0] == "S" else (G[b[1]] + G[b[2]])
    def sentence(b):
        row = counts(b).tocoo(); order = np.argsort(-row.data)[:a.kg]
        return " ".join(genes[row.col[j]] for j in order)
    def progs(b):
        row = counts(b).tocoo(); expr = set(int(c) for c, v in zip(row.col, row.data) if v > 0)
        out = []
        for lin in lins:
            hit = [m for m in RNA_MARK[lin] if m in gset and gi[m] in expr]
            if len(hit) >= a.min_mark: out.append((lin, hit))
        return out
    def scan_text(b):
        p = progs(b)
        return "; ".join(f"{l}-cell program: {', '.join(h)} [present]" for l, h in p) if p else "(none)"
    HEAD = ("Below is ONE droplet from a single-cell CITE-seq experiment. A droplet usually contains one cell (SINGLET) "
            "but sometimes two (DOUBLET). Lineages are T, B, NK, Mono. Decide SINGLET or DOUBLET, and state which lineage(s) "
            "are present. Answer EXACTLY two lines:\n'Final Answer: SINGLET' or 'Final Answer: DOUBLET'\n"
            "'Lineages: <comma-separated subset of T,B,NK,Mono>'")
    MARK_NOTE = ("Note: a single cell expresses ONE lineage program; if TWO mutually-exclusive programs are present it is likely a doublet. "
                 "The scan reports detectable lineage programs (including low-ranked markers).")
    def build(k, arm):
        b = bars[k]; gp = f"\nGenes (descending expression): {sentence(b)}"
        if arm == "l0": return HEAD + gp
        if arm == "ccmark": return HEAD + "\n" + MARK_NOTE + gp + f"\nLineage-program scan: {scan_text(b)}"
        raise ValueError(arm)
    def parse(t):
        if not t: return None, set()
        m = re.search(r"Final Answer:\s*(SINGLET|DOUBLET)", t, re.I)
        pred = m.group(1).upper() if m else (("DOUBLET" if t.upper().rfind("DOUBLET") > t.upper().rfind("SINGLET") else "SINGLET"))
        lm = re.search(r"Lineages?:\s*(.+)", t, re.I); seg = (lm.group(1) if lm else t).upper()
        named = {l for l in LINS_ALL if re.search(rf"\b{l.upper()}\b", seg) or (l == "Mono" and "MONO" in seg)}
        return pred, named
    client, mn = load_model(a.model, a.base_url); arms = a.arms.split(",")
    tasks = [(k, arm, build(k, arm)) for k in range(len(bars)) for arm in arms]
    print(f"model={a.model} bars={len(bars)} calls={len(tasks)}", flush=True)
    res = {}
    def call(t):
        k, arm, u = t
        try:
            kw = dict(model=mn, messages=[{"role":"system","content":SYS},{"role":"user","content":u}], max_tokens=a.max_tokens, temperature=a.temperature)
            if a.reasoning_effort: kw["reasoning_effort"] = a.reasoning_effort
            r = client.chat.completions.create(**kw); return (k, arm, parse(r.choices[0].message.content))
        except Exception as e: return (k, arm, ("__ERR__", set()))
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call, t) for t in tasks]):
            k, arm, pr = fut.result(); res[(k, arm)] = pr
    print(f"\n=== doublet 解释质量 (n={len(bars)}) ===")
    summary = {}
    for arm in arms:
        det = sum(res.get((k, arm), (None,))[0] == truth[k] for k in range(len(bars)))
        # explanation: among true doublets correctly called DOUBLET, named BOTH true lineages
        expl_ok = expl_tot = 0
        for k in range(len(bars)):
            if truth[k] != "DOUBLET": continue
            pred, named = res.get((k, arm), (None, set()))
            if pred != "DOUBLET": continue
            expl_tot += 1; expl_ok += int(set(true_lins[k]).issubset(named))
        det_acc = det / len(bars) * 100; expl_acc = (expl_ok / expl_tot * 100) if expl_tot else 0
        summary[arm] = {"det_acc": det_acc, "expl_acc": expl_acc, "expl_n": expl_tot}
        print(f"  {arm:8s} detection_acc={det_acc:5.1f}%  explanation_acc(named both true lineages)={expl_acc:5.1f}% (n={expl_tot})")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"model": a.model, "n": len(bars), "summary": summary,
               "preds": {f"{k}|{arm}": res[(k, arm)][0] for (k, arm) in res},
               "truth": {str(k): truth[k] for k in range(len(bars))}}, open(f"{a.out}/explain_{a.model}.json", "w"))
    print("saved")

if __name__ == "__main__": main()
