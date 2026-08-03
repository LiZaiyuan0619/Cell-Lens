"""SC-Arena CTA 的 CellCode A/B (开放式 CTA, 确定性本体距离评分, 支持 reasoning + 多端点)。
arms:
  L0       : 官方模板 (raw cell_sentence)
  cellcode : present-标记标注 — 对 cell 的 top 基因中的已知标记, 显式标注其指向的细胞类型
             (基因→类型字典来自 CellPuzzles 挖的标记, 标准符号, 跨数据集参考图谱)
评分: 确定性 CL 本体图距离 (复用 rescore 逻辑), 不用 LLM judge。
用法: python scarena_cellcode.py --model deepseek-v4-flash --arm cellcode --reasoning_effort high --n 608 --out results_scarena/cc_ds_cellcode_high.json
"""
import json, os, re, sys, time, argparse, threading, importlib.util, collections
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import networkx as nx
from cellpuzzles_rneg_oracle import mine_oracle, parse_puzzle  # 复用标记挖掘

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCARENA = os.path.join(HERE, "SC-Arena_repo")
DATA = os.path.join(PACKAGE_ROOT, "data", "public_benchmark_inputs", "scarena_cell_sentences_fixed.jsonl")
CP = os.path.join(PACKAGE_ROOT, "data", "public_benchmark_inputs", "cellpuzzles_test.jsonl")
EPS = {"endpoint1": "https://api.example.com/v1", "endpoint2": "https://api.example.com/v1",
       "endpoint3": "https://api.example.com/v1"}
TEMPLATE = ("The following is a list of {num_genes} gene names ordered by descending "
            "expression level in a {organism} cell.\nCell sentence: {cell_sentence}\n"
            "Your task is to identify the most likely cell type. Your response must "
            "include the answer in the following format: [Predicted_Cell_Type: ...]")
PAT = r"\[Predicted_Cell_Type:\s*([^\]]+?)\]"
DIST2SCORE = {0: 1.0, 1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}


def load_key():
    return os.environ.get("LLM_API_KEY")


# 手工 curated 经典标记面板 (标准符号, 含泛谱+亚型, 覆盖 SC-Arena 主要 lineage)
CURATED = {
    "T cell": ["CD3D", "CD3E", "CD3G", "TRAC", "TRBC2", "CD2"],
    "CD4-positive, alpha-beta T cell": ["CD4", "IL7R", "CD3D"],
    "CD8-positive, alpha-beta T cell": ["CD8A", "CD8B", "CD3D"],
    "naive T cell": ["CCR7", "SELL", "TCF7", "LEF1", "IL7R"],
    "central memory T cell": ["CCR7", "SELL", "IL7R", "GZMK"],
    "effector memory T cell": ["GZMK", "GZMA", "CCL5", "KLRG1"],
    "cytotoxic T cell": ["GZMB", "GZMH", "PRF1", "NKG7", "GNLY"],
    "regulatory T cell": ["FOXP3", "IL2RA", "CTLA4", "IKZF2"],
    "mucosal invariant T cell": ["SLC4A10", "KLRB1", "TRAV1-2"],
    "gamma-delta T cell": ["TRDC", "TRGC1", "TRGC2"],
    "natural killer cell": ["NKG7", "GNLY", "KLRD1", "NCAM1", "KLRF1", "NCR1"],
    "B cell": ["CD19", "MS4A1", "CD79A", "CD79B", "CD20"],
    "naive B cell": ["TCL1A", "IGHD", "IGHM", "FCER2"],
    "memory B cell": ["CD27", "TNFRSF13B", "MS4A1"],
    "plasma cell": ["MZB1", "XBP1", "JCHAIN", "IGHG1", "SDC1", "PRDM1"],
    "classical monocyte": ["CD14", "S100A8", "S100A9", "LYZ", "FCN1", "VCAN"],
    "non-classical monocyte": ["FCGR3A", "MS4A7", "CDKN1C", "LST1"],
    "macrophage": ["CD68", "C1QA", "C1QB", "C1QC", "APOE", "MRC1", "MARCO"],
    "alveolar macrophage": ["MARCO", "FABP4", "MCEMP1", "PPARG"],
    "conventional dendritic cell": ["FCER1A", "CD1C", "CLEC10A", "CLEC9A"],
    "plasmacytoid dendritic cell": ["LILRA4", "GZMB", "IL3RA", "CLEC4C", "IRF7"],
    "mast cell": ["TPSAB1", "TPSB2", "CPA3", "KIT", "MS4A2"],
    "neutrophil": ["FCGR3B", "CSF3R", "S100A8", "CXCR2"],
    "erythrocyte": ["HBA1", "HBA2", "HBB", "ALAS2"],
    "platelet": ["PPBP", "PF4", "GP9", "ITGA2B"],
    "endothelial cell": ["PECAM1", "VWF", "CLDN5", "CDH5", "FLT1", "KDR"],
    "vein endothelial cell": ["ACKR1", "SELP", "VWF"],
    "lymphatic endothelial cell": ["PROX1", "LYVE1", "PDPN", "FLT4"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRB", "COL3A1"],
    "smooth muscle cell": ["ACTA2", "MYH11", "TAGLN", "CNN1", "MYL9", "DES"],
    "pericyte": ["RGS5", "PDGFRB", "NOTCH3", "ACTA2"],
    "epithelial cell": ["EPCAM", "KRT8", "KRT18", "KRT19", "CDH1"],
    "epithelial cell of lung": ["SFTPB", "SFTPC", "SCGB1A1", "SCGB3A1", "NAPSA"],
    "type II pneumocyte": ["SFTPC", "SFTPB", "SFTPA1", "ABCA3"],
    "ciliated cell": ["FOXJ1", "TPPP3", "PIFO", "CAPS"],
    "hepatocyte": ["ALB", "APOA1", "TTR", "TF"],
    "enterocyte": ["FABP1", "FABP2", "APOA4", "RBP2"],
    "malignant cell": ["MKI67", "TOP2A", "EPCAM", "KRT8"],
}


def build_gene2types_curated(max_per_gene=4):
    g2t = collections.defaultdict(list)
    for t, mks in CURATED.items():
        for g in mks:
            g2t[g].append(t)
    return {g: ts[:max_per_gene] for g, ts in g2t.items()}


def build_gene2types(max_per_gene=3, max_types_specificity=8):
    """从 CellPuzzles 挖 type->markers, 反转成 gene->[types](只保留特异基因: 标记≤N个类型)。"""
    cp = [json.loads(l) for l in open(CP)]
    name2mk, _ = mine_oracle(cp, max_markers=15, topk_pool=40)  # type -> 富集 markers (有序)
    gene2types = collections.defaultdict(list)
    for t, mks in name2mk.items():
        for rank, g in enumerate(mks):
            gene2types[g].append((rank, t))
    out = {}
    for g, lst in gene2types.items():
        if len(lst) > max_types_specificity:      # 非特异(标记太多类型)→丢弃
            continue
        lst.sort()                                  # 按在各类型中的 rank
        out[g] = [t for _, t in lst[:max_per_gene]]
    return out


def build_type2markers(markers_src):
    """type -> markers 列表。curated=手工面板; cellpuzzles=从CP挖。"""
    if markers_src == "curated":
        return dict(CURATED)
    cp = [json.loads(l) for l in open(CP)]
    name2mk, _ = mine_oracle(cp, max_markers=15, topk_pool=40)
    return name2mk


def make_cellcode_prompt_gated(cell, type2markers, topk=30, min_hits=2):
    """类型级置信门控: 只在某类型≥min_hits个标记落在细胞top-K时提示; 都不达标则不显示块。"""
    genes = cell["cell_sentence"].split()
    topset = set(genes[:topk])
    ev = []
    for t, mks in type2markers.items():
        present = [m for m in mks if m in topset]
        if len(present) >= min_hits:
            ev.append((len(present), f"  - {t}: {', '.join(present)}"))
    ev.sort(reverse=True)                       # 命中多的在前
    base = TEMPLATE.format(num_genes=len(genes), organism=cell["tissue"],
                           cell_sentence=cell["cell_sentence"])
    if not ev:
        return base                              # 无置信证据 → 不误导, 退回 L0
    block = ("\nReference (supporting hints, not authoritative): among this cell's most highly-expressed "
             "genes, the following canonical cell-type markers were detected, grouped by the cell type they "
             "support (a type shown only if >=%d of its markers are top-expressed). Weigh these against the "
             "full expression profile and give the MOST SPECIFIC cell type:\n" % min_hits
             + "\n".join(s for _, s in ev[:8]))
    return base.replace("\nYour task is to identify", block + "\nYour task is to identify")


def make_cellcode_prompt(cell, gene2types):
    genes = cell["cell_sentence"].split()
    hits = []
    for g in genes:                                 # 保序(按表达降序)
        if g in gene2types:
            hits.append(f"  {g} → {', '.join(gene2types[g])}")
    base = TEMPLATE.format(num_genes=len(genes), organism=cell["tissue"],
                           cell_sentence=cell["cell_sentence"])
    if not hits:
        return base
    block = ("\nAmong the highly-expressed genes above, the following are known cell-type marker "
             "genes (gene → cell types it characteristically marks). Use these marker associations "
             "as evidence for your annotation:\n" + "\n".join(hits[:40]))
    # 插在 task 指令前
    return base.replace("\nYour task is to identify", block + "\nYour task is to identify")


def load_evaluator():
    if SCARENA not in sys.path:
        sys.path.insert(0, SCARENA)
    spec = importlib.util.spec_from_file_location("cta_eval",
            os.path.join(SCARENA, "evaluators", "cell_type_annotation.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.CellTypeEvaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="cellcode", choices=["L0", "cellcode"])
    ap.add_argument("--markers", default="cellpuzzles", choices=["cellpuzzles", "curated"])
    ap.add_argument("--evmode", default="flat", choices=["flat", "gated"], help="gated=类型级置信门控")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--shuffle", action="store_true", help="打乱映射(控制: 应砸穿)")
    ap.add_argument("--reasoning_effort", default="")
    ap.add_argument("--n", type=int, default=608)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--endpoints", default="hk,global,endpoint3")
    ap.add_argument("--max_tokens", type=int, default=8192)
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
        clients = [OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or EPS.get(e.strip(), e.strip()), timeout=600, max_retries=1)
                   for e in a.endpoints.split(",") if e.strip()]
    cells = [json.loads(l) for l in open(DATA)][: a.n]
    gene2types = {}; type2markers = {}
    if a.arm == "cellcode":
        if a.evmode == "gated":
            type2markers = build_type2markers(a.markers)
            if a.shuffle:
                ks = sorted(type2markers)
                type2markers = {ks[i]: type2markers[ks[(i + 1) % len(ks)]] for i in range(len(ks))}
                print("SHUFFLE: type->markers 错位映射", flush=True)
        else:
            gene2types = build_gene2types_curated() if a.markers == "curated" else build_gene2types()
            if a.shuffle:
                ks = sorted(gene2types)
                gene2types = {ks[i]: gene2types[ks[(i + 1) % len(ks)]] for i in range(len(ks))}
                print("SHUFFLE: gene->types 错位映射", flush=True)

    def base_of(c):
        return TEMPLATE.format(num_genes=len(c["cell_sentence"].split()),
                               organism=c["tissue"], cell_sentence=c["cell_sentence"])
    prompts = []; nblock = 0
    for c in cells:
        if a.arm != "cellcode":
            prompts.append(base_of(c))
        elif a.evmode == "gated":
            p = make_cellcode_prompt_gated(c, type2markers, a.topk, a.min_hits)
            prompts.append(p); nblock += (p != base_of(c))
        else:
            prompts.append(make_cellcode_prompt(c, gene2types))
    if a.arm == "cellcode":
        print(f"markers={a.markers} evmode={a.evmode}; cells WITH evidence block: {nblock}/{len(cells)}", flush=True)

    def call(pair):
        idx, p = pair
        client = clients[idx % len(clients)]
        for attempt in range(4):
            try:
                kw = dict(model=model_name, messages=[{"role": "user", "content": p}],
                          temperature=0.0, max_tokens=a.max_tokens, timeout=600)
                if a.reasoning_effort:
                    kw["reasoning_effort"] = a.reasoning_effort
                r = client.chat.completions.create(**kw)
                return idx, (r.choices[0].message.content or "")
            except Exception as e:
                last = f"{type(e).__name__}:{str(e)[:60]}"
                time.sleep(2 * (attempt + 1))
        return idx, f"[[ERROR {last}]]"

    answers = [None] * len(prompts); t0 = time.time(); done = 0; lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call, (i, p)) for i, p in enumerate(prompts)]):
            idx, resp = fut.result(); answers[idx] = resp
            with lock:
                done += 1
                if done % 50 == 0 or done == len(prompts):
                    print(f"{done}/{len(prompts)} [{time.time()-t0:.0f}s]", flush=True)
    errs = sum(1 for x in answers if x.startswith("[[ERROR"))

    # 确定性本体距离评分
    Eval = load_evaluator()
    ev = Eval(DATA, "http://x", "x", "none", evaluated_model="cc_rescore")
    UG = ev.graph.to_undirected()
    name2id = {}
    for t in ev.ontology.terms():
        if t.id.startswith("CL:") and t.name:
            name2id.setdefault(t.name.lower(), t.id)

    def node_of(name):
        n = name.strip().lower()
        if n in name2id:
            return name2id[n]
        for k, v in name2id.items():
            if n and (n in k or k in n):
                return v
        return None

    def det_score(pred, true):
        if pred.lower() == true.lower():
            return 1.0
        x, y = node_of(pred), node_of(true)
        if not x or not y or x not in UG or y not in UG:
            return 0.0
        try:
            d = nx.shortest_path_length(UG, x, y)
        except nx.NetworkXNoPath:
            return 0.0
        return DIST2SCORE.get(d, 0.0)

    tot = exct = 0.0
    for i, c in enumerate(cells):
        m = re.search(PAT, answers[i]); pred = m.group(1).strip() if m else "Unknown"
        true = c["cell_type"]
        tot += det_score(pred, true)
        if pred.lower() == true.lower():
            exct += 1
    n = len(cells)
    print(f"SC-Arena CTA [{a.model} {a.arm} re={a.reasoning_effort or 'none'}]: "
          f"det-acc={100*tot/n:.2f}%  exact={int(exct)}/{n}  errs={errs}  ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"model": a.model, "arm": a.arm, "reasoning": a.reasoning_effort,
               "det_acc": tot / n, "exact": int(exct), "n": n, "errs": errs,
               "answers": answers, "labels": [c["cell_type"] for c in cells]},
              open(a.out, "w"), indent=1)
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
