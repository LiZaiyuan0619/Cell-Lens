#!/usr/bin/env python3
"""增殖检测 A/B（WS2 新战场, 零下载）。cycle-program-high vs low, 冻结 LLM 读 cell-sentence。
立意: 周期程序(S/G2M)被 rank-only top-K 彻底压下(实测仅3%在top-200, 中位排名~1228)→表示遮蔽→CellCode摊开。
GT(确定性): CP10k+log1p 后 S/G2M panel 均值, combined=max(S,G2M); 增殖>=p90, 静息<=p60。诚实='周期程序高vs低'。
臂: l0 / ccmark(摊开被埋的S/G2M程序,flavor②) / ccshuf(程序块取自另一barcode,控制→应砸穿)
用法: .venv/bin/python proliferation_ab.py --model deepseek_v4_flash --n 60 --arms l0,ccmark,ccshuf
      .venv/bin/python proliferation_ab.py --dry
"""
import os, json, re, argparse, yaml
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

NPZ = "raw_data/mimitou_cite/cite_labeled.npz"
SYS = "You are an expert in single cell biology and genomics."
S_GENES = ['MCM5','PCNA','TYMS','FEN1','MCM2','MCM4','RRM1','UNG','GINS2','MCM6','CDCA7','DTL','PRIM1','UHRF1','HELLS','RFC2','RPA2','NASP','RAD51AP1','GMNN','SLBP','CCNE2','UBR7','POLD3','MSH2','ATAD2','RAD51','RRM2','CDC45','CDC6','EXO1','TIPIN','DSCC1','BLM','USP1','CLSPN','POLA1','CHAF1B','BRIP1','E2F8']
G2M_GENES = ['HMGB2','CDK1','NUSAP1','UBE2C','BIRC5','TPX2','TOP2A','NDC80','CKS2','NUF2','CKS1B','MKI67','TMPO','CENPF','TACC3','SMC4','CCNB2','CKAP2L','CKAP2','AURKB','BUB1','KIF11','ANP32E','TUBB4B','GTSE1','KIF20B','HJURP','CDCA3','CDC20','TTK','CDC25C','KIF2C','RANGAP1','NCAPD2','DLGAP5','CDCA2','CDCA8','ECT2','KIF23','HMMR','AURKA','PSRC1','ANLN','LBR','CKAP5','CENPE','NEK2']
EPS = {"endpoint1":"https://api.example.com/v1","endpoint2":"https://api.example.com/v1","endpoint3":"https://api.example.com/v1"}
def load_model(name, base_url=""):
    y = yaml.safe_load(open("AtomWorldBench/src/config/models.yaml"))[name]
    return OpenAI(api_key=os.environ.get(y["api_key"].strip("${}")), base_url=os.environ.get("LLM_BASE_URL") or EPS.get(base_url, base_url) or y["base_url"], timeout=180, max_retries=2), y["model_name"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek_v4_flash")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--kg", type=int, default=200)
    ap.add_argument("--arms", default="l0,ccmark,ccshuf")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--base_url", default="endpoint3")
    ap.add_argument("--reasoning_effort", default="")
    ap.add_argument("--npz", default=NPZ)
    ap.add_argument("--out", default="results_prolif")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=True)
    G = sparse.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"])).tocsr()
    genes = list(d["genes"]); gidx = {g: k for k, g in enumerate(genes)}
    Si = [gidx[g] for g in S_GENES if g in gidx]; Gi = [gidx[g] for g in G2M_GENES if g in gidx]
    tot = np.asarray(G.sum(1)).ravel(); tot[tot == 0] = 1
    Nm = G.multiply(1e4 / tot[:, None]).log1p().tocsr()
    Ssc = np.asarray(Nm[:, Si].mean(1)).ravel(); Gsc = np.asarray(Nm[:, Gi].mean(1)).ravel()
    comb = np.maximum(Ssc, Gsc)
    p90, p60 = np.percentile(comb, 90), np.percentile(comb, 60)
    rng = np.random.default_rng(a.seed)
    prolif = np.where(comb >= p90)[0]; rest = np.where(comb <= p60)[0]
    pi = rng.choice(prolif, size=min(a.n, len(prolif)), replace=False)
    ri = rng.choice(rest, size=min(a.n, len(rest)), replace=False)
    cells = [("CYCLING", int(i)) for i in pi] + [("RESTING", int(i)) for i in ri]
    rng.shuffle(cells); truth = [c[0] for c in cells]
    print(f"model={a.model} cells={len(cells)} (cycling={len(pi)} resting={len(ri)}) p90={p90:.3f}", flush=True)
    def rna_sentence(i):
        row = G[i].tocoo(); order = np.argsort(-row.data)[:a.kg]
        return " ".join(genes[row.col[j]] for j in order)
    def cycle_scan(i):
        coo = G[i].tocoo(); expr = set(int(c) for c, v in zip(coo.col, coo.data) if v > 0)
        return [genes[j] for j in Si if j in expr], [genes[j] for j in Gi if j in expr]
    permmap = list(range(len(cells))); rng.shuffle(permmap)
    HEAD = ("Below is the data for ONE single cell. Decide whether this cell is actively in the CELL CYCLE "
            "(proliferating: S-phase or G2M program ON) or RESTING (quiescent/G0-G1, cycle program OFF). "
            "Answer 'Final Answer: CYCLING' or 'Final Answer: RESTING'.")
    SPEC = ("Note: the gene list shows only top-expressed genes; cell-cycle genes are expressed at modest levels and "
            "get pushed far below the top, so they are hard to see in the list.")
    NOTE = ("Note: the scan below COUNTS how many cell-cycle marker genes are detected in this cell (these are usually "
            "ranked too low to appear in the gene list above). A cycling cell expresses MANY cell-cycle markers (high "
            "counts across S-phase and G2M); a resting cell expresses FEW. Judge mainly from the COUNTS.")
    def mk(i):
        sp, gp = cycle_scan(i)
        return (f"S-phase cell-cycle markers detected: {len(sp)} of {len(Si)} ({', '.join(sp[:8])}{'...' if len(sp)>8 else ''}); "
                f"G2M markers detected: {len(gp)} of {len(Gi)} ({', '.join(gp[:8])}{'...' if len(gp)>8 else ''})")
    def build(k, arm):
        i = cells[k][1]; gpart = f"\nGenes (descending expression): {rna_sentence(i)}"
        if arm == "l0":     return HEAD + gpart
        if arm == "ccmark": return HEAD + "\n" + SPEC + "\n" + NOTE + gpart + f"\nCell-cycle scan: {mk(i)}"
        if arm == "ccshuf": return HEAD + "\n" + SPEC + "\n" + NOTE + gpart + f"\nCell-cycle scan: {mk(cells[permmap[k]][1])}"
        raise ValueError(arm)
    def parse(t):
        if not t: return None
        m = re.search(r"Final Answer:\s*(.+)", t, re.I); seg = (m.group(1) if m else t).upper()
        h = [(seg.rfind("CYCLING"), "CYCLING"), (seg.rfind("RESTING"), "RESTING")]; h = [(p, w) for p, w in h if p >= 0]
        return sorted(h)[-1][1] if h else None
    if a.dry:
        import collections
        for tv in ["CYCLING", "RESTING"]:
            cnt = collections.Counter()
            for k in range(len(cells)):
                if truth[k] == tv:
                    sp, gp = cycle_scan(cells[k][1]); cnt[len(sp) + len(gp)] += 1
            print(f"{tv}: #检出周期marker分布(前10) = {dict(sorted(cnt.items())[:10])}  中位={int(np.median([len(cycle_scan(cells[k][1])[0])+len(cycle_scan(cells[k][1])[1]) for k in range(len(cells)) if truth[k]==tv]))}")
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
    print(f"\n=== 增殖检测 (n={len(cells)}: cycling={len(pi)} resting={len(ri)}) ===")
    acc = {}
    for arm in arms:
        ok = sum(res.get((k, arm)) == truth[k] for k in range(len(cells)))
        cy = sum(res.get((k, arm)) == "CYCLING" for k in range(len(cells)) if truth[k] == "CYCLING")
        rr = sum(res.get((k, arm)) == "RESTING" for k in range(len(cells)) if truth[k] == "RESTING")
        err = sum(str(res.get((k, arm))).startswith("__ERR__") for k in range(len(cells)))
        acc[arm] = ok / len(cells)
        print(f"  {arm:8s} acc={ok/len(cells)*100:5.1f}%  cycling-recall={cy/len(pi)*100:5.1f}%  resting-recall={rr/len(ri)*100:5.1f}%  err={err}")
    if "l0" in arms:
        base = [res.get((k, "l0")) == truth[k] for k in range(len(cells))]
        for arm in arms:
            if arm == "l0": continue
            oth = [res.get((k, arm)) == truth[k] for k in range(len(cells))]
            b01 = sum(1 for x, y in zip(base, oth) if (not x) and y); b10 = sum(1 for x, y in zip(base, oth) if x and (not y))
            print(f"  l0 vs {arm}: gain(+)={b01} loss(-)={b10} net={b01-b10:+d}")
    os.makedirs(a.out, exist_ok=True)
    json.dump({"acc": acc, "model": a.model, "n": len(cells), "arms": arms,
               "preds": {f"{k}|{arm}": res[(k, arm)] for (k, arm) in res}, "truth": {str(k): truth[k] for k in range(len(cells))}}, open(f"{a.out}/prolif_{a.model}.json", "w"))
    print("saved")

if __name__ == "__main__": main()
