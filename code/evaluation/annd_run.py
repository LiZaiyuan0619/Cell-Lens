"""AnnDictionary 式探针 Step2: de-novo cluster CTA 的 L0 vs CellCode(flavor② 标记标注) A/B。
L0       : 裸 de-novo DE 基因表 → CTA
cellcode : + 门控标记证据(某候选类型≥2标记命中才提示) → 帮在噪声里定型
评分: lineage 级(T/B/NK/Mono) + full 级(含亚型 naive/effector,CD4/CD8)。
用法: python annd_run.py --model deepseek-v4-flash --arm cellcode --reasoning_effort high --clusters new_benchmarks/annd_tenx_clusters.json
"""
import json, os, re, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from scarena_cellcode import build_type2markers   # 复用 CURATED 标记面板

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EPS = {"endpoint1": "https://api.example.com/v1", "endpoint2": "https://api.example.com/v1",
       "endpoint3": "https://api.example.com/v1"}


def load_key():
    return os.environ.get("LLM_API_KEY")


def gated_markers(top_genes, t2m, topk=15, min_hits=2):
    topset = set(top_genes[:topk]); ev = []
    for t, mks in t2m.items():
        present = [m for m in mks if m in topset]
        if len(present) >= min_hits:
            ev.append((len(present), f"  - {t}: {', '.join(present)}"))
    ev.sort(reverse=True)
    return [s for _, s in ev[:8]]


def make_prompt(cl, arm, t2m, block_src=None):
    """block_src: 哪个簇的标记块贴上来 (默认本簇=cl). shuffle 时传别的簇 -> 错位标注控制。"""
    g = ", ".join(cl["top_genes"])
    base = (f"The following are the top differentially-expressed marker genes of a de-novo "
            f"single-cell cluster (PBMC), ranked by enrichment:\n{g}\n"
            f"Identify the most likely cell type. Respond in the format: [Cell_Type: ...]")
    if arm == "L0":
        return base
    src = block_src if block_src is not None else cl
    ev = gated_markers(src["top_genes"], t2m)  # 用 src 的 top_genes 算标记块 (shuffle: src≠cl)
    if not ev:
        return base
    block = ("\nReference (supporting hints): among the genes above, these are canonical markers, "
             "grouped by the cell type they support (a type shown only if >=2 of its markers appear). "
             "Weigh against the full list and give the MOST SPECIFIC type:\n" + "\n".join(ev))
    return base.replace("\nIdentify the most likely", block + "\nIdentify the most likely")


def make_prompt_f3(cl, pblock):
    """f3 arm (flavor 1): attach the cluster's mean surface-protein signature (orthogonal modality)."""
    g = ", ".join(cl["top_genes"])
    base = (f"The following are the top differentially-expressed marker genes of a de-novo "
            f"single-cell cluster (PBMC), ranked by enrichment:\n{g}\n"
            f"Identify the most likely cell type. Respond in the format: [Cell_Type: ...]")
    block = ("\nReference: mean surface-protein measurements for this cluster (CLR-normalized; an "
             "orthogonal modality not derivable from the gene list above; higher = more abundant):\n  " + pblock)
    return base.replace("\nIdentify the most likely", block + "\nIdentify the most likely")


# ----- 评分: 归一化到 (lineage, subtype) -----
def norm(label):
    l = label.lower()
    if any(k in l for k in ["monocyt", "cd14", "cd16", "macrophage"]): return ("Mono", "")
    if "nk" in l or "natural killer" in l: return ("NK", "")
    if " b " in f" {l} " or l.endswith(" b") or "b cell" in l or "plasma" in l: return ("B", "")
    if "t cell" in l or "cd4" in l or "cd8" in l or l.endswith(" t") or " t " in f" {l} ":
        sub = "CD4" if "cd4" in l else ("CD8" if "cd8" in l else "")
        nm = "naive" if ("naive" in l or "ccr7" in l) else ("eff" if ("effector" in l or "memory" in l or "cytotox" in l) else "")
        return ("T", f"{sub}_{nm}")
    if "unassign" in l or "unknown" in l: return ("Unassigned", "")
    return ("other", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="cellcode", choices=["L0", "cellcode", "shuffle", "f3", "f3shuf"])
    ap.add_argument("--protein", default=os.path.join(PACKAGE_ROOT, "data", "cellspectrum_examples", "annd_tenx_protein.json"),
                    help="f3 arm: per-cluster mean protein block json (from annd_protein_block.py)")
    ap.add_argument("--reasoning_effort", default="")
    ap.add_argument("--clusters", default=os.path.join(PACKAGE_ROOT, "data", "cellspectrum_examples", "annd_tenx_clusters.json"))
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # --- resolve model via models.yaml if --model is a yaml alias (alt flagships etc.) ---
    import os as _os, yaml as _yaml
    _cfg = _yaml.safe_load(open(_os.path.join(PACKAGE_ROOT,
                                              "AtomWorldBench/src/config/models.yaml")))
    model_name = a.model
    if a.model in _cfg:
        y = _cfg[a.model]
        model_name = y["model_name"]
        key = _os.environ.get(y["api_key"].strip("${}")) or load_key()
        clients = [OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or y["base_url"], timeout=600, max_retries=1)]
    else:
        key = load_key()
        clients = [OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or u, timeout=600, max_retries=1) for u in EPS.values()]
    cls = json.load(open(a.clusters))
    t2m = build_type2markers("curated") if a.arm in ("cellcode", "shuffle") else {}
    if a.arm in ("f3", "f3shuf"):
        prot = json.load(open(a.protein))
        n = len(cls)
        if a.arm == "f3":
            srcidx = list(range(n))
        else:  # f3shuf: derangement -> each cluster gets ANOTHER cluster's protein block
            import random as _r
            rng = _r.Random(a.seed); srcidx = list(range(n)); rng.shuffle(srcidx)
            for k in range(n):
                if srcidx[k] == k:
                    j = (k + 1) % n; srcidx[k], srcidx[j] = srcidx[j], srcidx[k]
        prompts = [make_prompt_f3(cls[k], prot[str(cls[srcidx[k]]["cluster"])]["protein_block"]) for k in range(n)]
        print(f"F3 arm={a.arm}: protein block on {n} clusters "
              f"(fixed-points={sum(srcidx[k]==k for k in range(n))})", flush=True)
    elif a.arm == "shuffle":
        # 错位标注控制: 每个簇贴的是 *别的* 簇的标记块 (derangement, 无固定点) → 测模型是否真读这个块。
        # 若 shuffle 也砸穿(≈cellcode 掉点) ⇒ 模型读了块且被误导 ⇒ −21.6 是真"方法施加"的边界, 非被忽略的输入。
        import random as _r
        rng = _r.Random(a.seed); n = len(cls)
        perm = list(range(n)); rng.shuffle(perm)
        for k in range(n):  # 消除固定点
            if perm[k] == k:
                j = (k + 1) % n; perm[k], perm[j] = perm[j], perm[k]
        prompts = [make_prompt(cls[k], "cellcode", t2m, block_src=cls[perm[k]]) for k in range(n)]
        nb = sum(1 for k in range(n) if prompts[k] != make_prompt(cls[k], "L0", t2m))
        print(f"SHUFFLE control: {nb}/{n} clusters got a (mismatched) marker block; "
              f"fixed-points={sum(perm[k]==k for k in range(n))}", flush=True)
    else:
        prompts = [make_prompt(c, a.arm, t2m) for c in cls]
        nb = sum(1 for p, c in zip(prompts, cls) if a.arm == "cellcode" and p != make_prompt(c, "L0", t2m))
        if a.arm == "cellcode":
            print(f"{nb}/{len(cls)} clusters got marker block", flush=True)

    def call(pair):
        i, p = pair; client = clients[i % len(clients)]
        for att in range(4):
            try:
                kw = dict(model=model_name, messages=[{"role": "user", "content": p}],
                          temperature=0.0, max_tokens=a.max_tokens, timeout=600)
                if a.reasoning_effort: kw["reasoning_effort"] = a.reasoning_effort
                r = client.chat.completions.create(**kw)
                return i, (r.choices[0].message.content or "")
            except Exception as e:
                last = f"{type(e).__name__}:{str(e)[:50]}"; time.sleep(2 * (att + 1))
        return i, f"[[ERR {last}]]"

    ans = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call, (i, p)) for i, p in enumerate(prompts)]):
            i, r = fut.result(); ans[i] = r

    lin = full = tot = 0.0
    rows = []
    for c, r in zip(cls, ans):
        m = re.search(r"\[Cell_Type:\s*([^\]]+?)\]", r); pred = m.group(1).strip() if m else "?"
        tl, ts = norm(c["true_label"]); pl, ps = norm(pred)
        lin_ok = (tl == pl and tl not in ("other", "Unassigned"))
        full_ok = lin_ok and (ts == ps)
        # weight by cluster size
        w = c["n"]; tot += w; lin += w * lin_ok; full += w * full_ok
        rows.append({"cluster": c["cluster"], "true": c["true_label"], "pred": pred,
                     "lin_ok": lin_ok, "full_ok": full_ok, "purity": c["purity"]})
    print(f"AnnD [{a.model} {a.arm} re={a.reasoning_effort or 'none'}]: "
          f"lineage-acc={100*lin/tot:.1f}%  full-acc={100*full/tot:.1f}%  (cell-weighted, {len(cls)} clusters)", flush=True)
    json.dump({"model": a.model, "arm": a.arm, "lineage_acc": lin/tot, "full_acc": full/tot,
               "rows": rows, "answers": ans}, open(a.out, "w"), indent=1)
    print(f"saved {a.out}", flush=True)
    for r in rows:
        print(f"  c{r['cluster']:2d} pur={r['purity']:.2f} true={r['true']:20s} pred={r['pred'][:30]:30s} "
              f"{'L✓' if r['lin_ok'] else 'L✗'}{'F✓' if r['full_ok'] else 'F✗'}")


if __name__ == "__main__":
    main()
