#!/usr/bin/env python3
"""高线粒体 QC 检测 A/B —— 四判据【负控制】(WS2, 零下载)。
HIGH-MITO(濒死/低质量) vs LOW-MITO, 冻结 LLM 读 cell-sentence。

立意(关键对照): 与增殖/doublet 相反, MT 基因表达极高 → 在 rank-only top-K 里**位于最前**
(实测高-mito 细胞 MT 基因中位排名~2, 100% 落 top-200) → 判别信号**未被埋、可读**。
四判据(规则A·判据②不可恢复)预测: 信号既然可读, CellCode 把 MT marker 摊开/计数 → **不应涨(≈持平)**。
ccmark 故意只'数 MT marker 检出'(不直接给 GT 的 MT 分数), 与 proliferation 同构操作; 因 MT 基因
近乎所有细胞都表达(计数无判别力) + 信号已在句子顶端 → 预测 ccmark≈l0。命中即'预测洗掉→果然洗掉'金子。

GT(确定性): MT 计数占比 = sum(MT-* counts)/total; HIGH>=p_hi, LOW<=p_lo。
臂: l0 / ccmark(数 MT marker 检出, flavor② 同构) / ccshuf(marker 块取自另一 barcode, 控制)
用法: .venv/bin/python highmito_ab.py --model deepseek_v4_flash --n 150 --arms l0,ccmark,ccshuf --base_url endpoint3
      .venv/bin/python highmito_ab.py --dry
"""
import os, json, re, argparse, yaml
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

NPZ = "raw_data/mimitou_cite/cite_labeled.npz"
SYS = "You are an expert in single cell biology and genomics."
EPS = {"endpoint1":"https://api.example.com/v1","endpoint2":"https://api.example.com/v1","endpoint3":"https://api.example.com/v1"}
def load_model(name, base_url=""):
    y = yaml.safe_load(open("AtomWorldBench/src/config/models.yaml"))[name]
    return OpenAI(api_key=os.environ.get(y["api_key"].strip("${}")), base_url=os.environ.get("LLM_BASE_URL") or EPS.get(base_url, base_url) or y["base_url"], timeout=180, max_retries=2), y["model_name"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek_v4_flash")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--kg", type=int, default=200)
    ap.add_argument("--p_hi", type=float, default=90)
    ap.add_argument("--p_lo", type=float, default=50)
    ap.add_argument("--arms", default="l0,ccmark,ccshuf")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--base_url", default="endpoint3")
    ap.add_argument("--reasoning_effort", default="")
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--out", default="results_highmito")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    G = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])).tocsr()
    genes = list(d["genes"]); gidx = {g: k for k, g in enumerate(genes)}
    MT = [g for g in genes if g.startswith("MT-")]; MTi = [gidx[g] for g in MT]
    tot = np.asarray(G.sum(1)).ravel(); tot[tot == 0] = 1
    mtfrac = np.asarray(G[:, MTi].sum(1)).ravel() / tot
    p_hi, p_lo = np.percentile(mtfrac, a.p_hi), np.percentile(mtfrac, a.p_lo)
    rng = np.random.default_rng(a.seed)
    hi = np.where(mtfrac >= p_hi)[0]; lo = np.where(mtfrac <= p_lo)[0]
    hii = rng.choice(hi, size=min(a.n, len(hi)), replace=False)
    loi = rng.choice(lo, size=min(a.n, len(lo)), replace=False)
    cells = [("HIGH-MITO", int(i)) for i in hii] + [("LOW-MITO", int(i)) for i in loi]
    rng.shuffle(cells); truth = [c[0] for c in cells]
    print(f"model={a.model} cells={len(cells)} (high={len(hii)} low={len(loi)}) MT%hi>={p_hi*100:.1f} lo<={p_lo*100:.1f}", flush=True)
    def rna_sentence(i):
        row = G[i].tocoo(); order = np.argsort(-row.data)[:a.kg]
        return " ".join(genes[row.col[j]] for j in order)
    def mt_scan(i):
        coo = G[i].tocoo(); expr = set(int(c) for c, v in zip(coo.col, coo.data) if v > 0)
        return [g for g, j in zip(MT, MTi) if j in expr]
    permmap = list(range(len(cells))); rng.shuffle(permmap)
    HEAD = ("Below is the data for ONE single cell. Decide whether this cell is HIGH-MITO (a high fraction of its "
            "transcripts come from mitochondrial genes, typical of a stressed/dying/low-quality cell) or LOW-MITO "
            "(a healthy cell with low mitochondrial fraction). Answer 'Final Answer: HIGH-MITO' or 'Final Answer: LOW-MITO'.")
    NOTE = ("Note: the scan below COUNTS how many mitochondrial (MT-) genes are detected in this cell. A HIGH-MITO cell "
            "has a large mitochondrial transcript fraction; a LOW-MITO cell has a small one. Judge from this scan plus the gene list.")
    def mk(i):
        sp = mt_scan(i)
        return f"Mitochondrial genes detected: {len(sp)} of {len(MT)} ({', '.join(sp[:13])})"
    def build(k, arm):
        i = cells[k][1]; gpart = f"\nGenes (descending expression): {rna_sentence(i)}"
        if arm == "l0":     return HEAD + gpart
        if arm == "ccmark": return HEAD + "\n" + NOTE + gpart + f"\nMitochondrial scan: {mk(i)}"
        if arm == "ccshuf": return HEAD + "\n" + NOTE + gpart + f"\nMitochondrial scan: {mk(cells[permmap[k]][1])}"
        raise ValueError(arm)
    def parse(t):
        if not t: return None
        m = re.search(r"Final Answer:\s*(.+)", t, re.I); seg = (m.group(1) if m else t).upper()
        h = [(seg.rfind("HIGH"), "HIGH-MITO"), (seg.rfind("LOW"), "LOW-MITO")]; h = [(p, w) for p, w in h if p >= 0]
        return sorted(h)[-1][1] if h else None
    if a.dry:
        import collections
        for tv in ["HIGH-MITO", "LOW-MITO"]:
            cnt = collections.Counter(len(mt_scan(cells[k][1])) for k in range(len(cells)) if truth[k] == tv)
            ranks = []
            for k in range(len(cells)):
                if truth[k] != tv: continue
                row = G[cells[k][1]].tocoo(); order = list(np.argsort(-row.data)); cols = [row.col[j] for j in order]
                rk = [cols.index(m)+1 for m in MTi if m in cols]
                if rk: ranks.append(min(rk))
            print(f"{tv}: #MT基因检出分布={dict(sorted(cnt.items()))} 顶MT基因中位排名={int(np.median(ranks))}")
        print("→ 若两类 #MT基因检出 高度重叠 = 计数无判别力(预期); MT 排名靠前 = 信号可读(负控制前提)")
        return
    client, mn = load_model(a.model, a.base_url); arms = a.arms.split(",")
    tasks = [(k, arm, build(k, arm)) for k in range(len(cells)) for arm in arms]
    print(f"calls={len(tasks)}", flush=True); res = {}
    def call(t):
        k, arm, u = t
        try:
            kw = dict(model=mn, messages=[{"role":"system","content":SYS},{"role":"user","content":u}], max_tokens=a.max_tokens, temperature=0)
            if a.reasoning_effort: kw["reasoning_effort"] = a.reasoning_effort
            r = client.chat.completions.create(**kw); return (k, arm, parse(r.choices[0].message.content))
        except Exception as e: return (k, arm, f"__ERR__{type(e).__name__}")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call, t) for t in tasks]):
            k, arm, p = fut.result(); res[(k, arm)] = p
    print(f"\n=== 高mito负控制 (n={len(cells)}: high={len(hii)} low={len(loi)}) — 预测 ccmark≈l0 ===")
    acc = {}
    for arm in arms:
        ok = sum(res.get((k, arm)) == truth[k] for k in range(len(cells)))
        hr = sum(res.get((k, arm)) == "HIGH-MITO" for k in range(len(cells)) if truth[k] == "HIGH-MITO")
        lr = sum(res.get((k, arm)) == "LOW-MITO" for k in range(len(cells)) if truth[k] == "LOW-MITO")
        err = sum(str(res.get((k, arm))).startswith("__ERR__") for k in range(len(cells)))
        acc[arm] = ok / len(cells)
        print(f"  {arm:8s} acc={ok/len(cells)*100:5.1f}%  high-recall={hr/len(hii)*100:5.1f}%  low-recall={lr/len(loi)*100:5.1f}%  err={err}")
    if "l0" in arms:
        base = [res.get((k, "l0")) == truth[k] for k in range(len(cells))]
        for arm in arms:
            if arm == "l0": continue
            oth = [res.get((k, arm)) == truth[k] for k in range(len(cells))]
            b01 = sum(1 for x, y in zip(base, oth) if (not x) and y); b10 = sum(1 for x, y in zip(base, oth) if x and (not y))
            print(f"  l0 vs {arm}: gain(+)={b01} loss(-)={b10} net={b01-b10:+d}")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"acc": acc, "model": a.model, "n": len(cells), "arms": arms,
               "preds": {f"{k}|{arm}": res[(k, arm)] for (k, arm) in res}, "truth": {str(k): truth[k] for k in range(len(cells))}}, open(f"{a.out}/highmito_{a.model}.json", "w"))
    print("saved")

if __name__ == "__main__": main()
