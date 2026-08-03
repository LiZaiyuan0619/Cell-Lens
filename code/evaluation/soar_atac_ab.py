#!/usr/bin/env python3
"""SOAR α smoke: ATAC-only cell typing — 'collapse into RNA-text' vs 'typed native-ATAC repair'.

Motivation: SOAR-MultiOmics feeds ATAC-measured cells to a frozen LLM by VAE-bridging
ATAC -> predicted-RNA -> top-10 gene NAMES (lossy: reconstruction + no values + truncation).
CellCode 'repairs' by giving the native ATAC gene-activity as an EXPLICIT TYPED block with
log-activity VALUES + more markers (no RNA, no VAE). Test: does the typed repair beat the
lossy collapse, using ONLY ATAC-derived info?  (Distinct from card-18, which added ATAC on
top of native RNA and was null because RNA already carried the signal.)

Arms (all ATAC-only, NO native RNA, NO VAE):
  collapse      = top-Kc ATAC gene-activity gene NAMES only, plain 'Genes: ...' (framed as expression) [SOAR-like lossy]
  typed         = top-Kt ATAC gene-activity genes WITH log-activity VALUES + explicit chromatin note   [alpha repair]
  collapse_shuf = collapse but another cell's ATAC (cross-cell permutation control)
  typed_shuf    = typed    but another cell's ATAC
usage: set -a && . ./.env && set +a && \
       .venv/bin/python code/evaluation/soar_atac_ab.py --model deepseek_v4_flash --n 100 --workers 16
"""
import os, json, re, time, argparse, yaml
import numpy as np
from scipy import sparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

NPZ = "raw_data/multiome_10k/multiome_labeled.npz"
SYS = ("You are an expert who knows a lot about single cell biology and genomics and will "
       "help me solve a series of tasks related to single cell data analysis.")
EPS = {"endpoint1": "https://api.example.com/v1", "endpoint2": "https://api.example.com/v1",
       "endpoint3": "https://api.example.com/v1"}

CHROM_NOTE = ("Note: this cell was profiled by scATAC-seq (chromatin accessibility), NOT RNA expression. "
              "Below are the genes with the most accessible chromatin (open promoter/gene body), listed as "
              "NAME:v where v = log(1 + ATAC gene-activity) and higher means more open chromatin at that "
              "gene's locus. Open chromatin at a lineage-defining gene indicates that gene is poised/active "
              "in this cell's lineage. Use these accessibility signals to infer the cell type.")

CHROM_NOTE_NAMES = ("Note: this cell was profiled by scATAC-seq (chromatin accessibility), NOT RNA expression. "
                    "Below are the genes with the most accessible chromatin (open promoter/gene body). "
                    "Open chromatin at a lineage-defining gene indicates that gene is poised/active "
                    "in this cell's lineage. Use these accessibility signals to infer the cell type.")


def load_model(name, base_url=""):
    y = yaml.safe_load(open("AtomWorldBench/src/config/models.yaml"))[name]
    key = os.environ.get(y["api_key"].strip("${}"))
    url = EPS.get(base_url, base_url) or os.environ.get("LLM_BASE_URL") or y["base_url"]
    return OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or url, timeout=180, max_retries=2), y["model_name"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek_v4_flash")
    ap.add_argument("--n", type=int, default=100, help="cells per class")
    ap.add_argument("--kc", type=int, default=10, help="collapse: #ATAC gene names (SOAR uses top-10)")
    ap.add_argument("--kt", type=int, default=30, help="typed: #ATAC genes shown with values")
    ap.add_argument("--arms", default="collapse,typed,collapse_shuf,typed_shuf")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--out", default="results_soar_atac")
    ap.add_argument("--base_url", default="endpoint3")
    ap.add_argument("--npz", default=NPZ)
    a = ap.parse_args()

    CANDS = ["CD4 T", "CD8 T", "B", "NK", "Monocyte"]
    cands_str = ", ".join(f"'{c}'" for c in CANDS) + "."

    d = np.load(a.npz, allow_pickle=True)
    GA = sparse.csr_matrix((d["ga_data"], d["ga_indices"], d["ga_indptr"]), shape=tuple(d["ga_shape"]))
    ga_genes = d["ga_genes"]
    labels = d["labels"]

    # TF-IDF on gene-activity (suppress housekeeping / promoter-dense bias) — same as multiome_ab.py
    N = GA.shape[0]
    df = np.asarray((GA > 0).sum(0)).ravel(); idf = np.log((N + 1) / (df + 1)) + 1
    rowsum = np.asarray(GA.sum(1)).ravel() + 1e-9
    GA = GA.tocsr()
    tf = GA.data / np.repeat(rowsum, np.diff(GA.indptr))                  # per-nonzero tf WITHOUT full-matrix .tocoo()
    TFIDF = sparse.csr_matrix((tf * idf[GA.indices], GA.indices.copy(), GA.indptr.copy()), shape=GA.shape)
    del tf, rowsum   # avoid tocoo → peak ~1.9GB→~0.9GB → 2 concurrent fits under ~5GB cgroup

    rng0 = np.random.default_rng(a.seed)
    idx_by = {}
    for i, l in enumerate(labels):
        if l in CANDS:
            idx_by.setdefault(l, []).append(i)
    use = []
    for l in CANDS:
        ids = idx_by.get(l, []); rng0.shuffle(ids); use += ids[:a.n]
    rng0.shuffle(use)
    print(f"model={a.model} cells={len(use)} per-class="
          f"{ {l: min(len(idx_by.get(l, [])), a.n) for l in CANDS} }", flush=True)

    def atac_names(i, k):  # top-k TF-IDF ATAC genes, names only
        row = TFIDF[i].tocoo()
        if row.nnz == 0:
            return []
        order = row.col[np.argsort(-row.data)[:k]]
        return [ga_genes[j] for j in order]

    def atac_valued(i, k):  # top-k TF-IDF ATAC genes, NAME:log-activity (log1p on-the-fly; no full GAlog copy)
        row = TFIDF[i].tocoo()
        if row.nnz == 0:
            return "(none)"
        order = row.col[np.argsort(-row.data)[:k]]
        gi = GA[i]  # sparse row of raw gene-activity
        return " ".join(f"{ga_genes[j]}:{np.log1p(gi[0, j]):.1f}" for j in order)

    names_c = {i: atac_names(i, a.kc) for i in use}
    names_t = {i: atac_names(i, a.kt) for i in use}
    valued_t = {i: atac_valued(i, a.kt) for i in use}

    perm = use.copy(); rng0.shuffle(perm); permmap = dict(zip(use, perm))

    HEAD = (f"Select the one cell type this single cell most likely belongs to, from the following "
            f"candidates: {cands_str} Make your choice in the format 'Final Answer: Prediction'.")

    def build(i, arm):
        if arm == "collapse":
            return HEAD + f"\nGenes: {' '.join(names_c[i])}"
        if arm == "collapse_shuf":
            return HEAD + f"\nGenes: {' '.join(names_c[permmap[i]])}"
        if arm == "typed":
            return HEAD + "\n" + CHROM_NOTE + f"\nAccessible genes (descending accessibility): {valued_t[i]}"
        if arm == "typed_shuf":
            return HEAD + "\n" + CHROM_NOTE + f"\nAccessible genes (descending accessibility): {valued_t[permmap[i]]}"
        if arm == "names30":       # decompose: collapse's plain framing but top-kt NAMES (isolate gene-count 10->30)
            return HEAD + f"\nGenes: {' '.join(names_t[i])}"
        if arm == "names30_note":  # + chromatin-typing context, still names-only (isolate modality-typing effect)
            return HEAD + "\n" + CHROM_NOTE_NAMES + f"\nAccessible genes (descending accessibility): {' '.join(names_t[i])}"
        raise ValueError(arm)

    def parse_pred(text):
        if not text:
            return None
        m = re.search(r"Final Answer:\s*(.+)", text, re.I)
        seg = (m.group(1) if m else "").strip().strip("'\"`*. ").lower()
        if seg:
            for c in CANDS:
                if seg == c.lower():
                    return c
            for c in sorted(CANDS, key=len, reverse=True):
                if c.lower() in seg or seg in c.lower():
                    return c
        low = text.lower(); hits = [(low.rfind(c.lower()), c) for c in CANDS if c.lower() in low]
        if hits:
            return sorted(hits, key=lambda h: (h[0], len(h[1])))[-1][1]
        return None

    client, model_name = load_model(a.model, a.base_url)
    arms = a.arms.split(",")
    tasks = [(i, arm, build(i, arm)) for i in use for arm in arms]
    print(f"total LLM calls={len(tasks)}", flush=True)
    results = {}

    def call(t):
        i, arm, u = t
        try:
            r = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": u}],
                max_tokens=a.max_tokens, temperature=0)
            return (i, arm, parse_pred(r.choices[0].message.content))
        except Exception as e:
            return (i, arm, f"__ERR__{type(e).__name__}")

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for fut in as_completed([ex.submit(call, t) for t in tasks]):
            i, arm, pred = fut.result(); results[(i, arm)] = pred; done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(tasks)} [{time.time() - t0:.0f}s]", flush=True)

    print(f"\n=== accuracy (n={len(use)} cells, ATAC-only) ===")
    acc = {}
    for arm in arms:
        ok = sum(results.get((i, arm)) == labels[i] for i in use)
        err = sum(str(results.get((i, arm))).startswith("__ERR__") for i in use)
        acc[arm] = ok / len(use)
        print(f"  {arm:14s} acc={ok / len(use) * 100:5.1f}%  ({ok}/{len(use)})  err={err}")
    base = "collapse"
    if base in arms:
        b = [results.get((i, base)) == labels[i] for i in use]
        for arm in arms:
            if arm == base:
                continue
            o = [results.get((i, arm)) == labels[i] for i in use]
            b01 = sum(1 for x, y in zip(b, o) if (not x) and y)
            b10 = sum(1 for x, y in zip(b, o) if x and (not y))
            print(f"  collapse vs {arm:14s}: gain(+)={b01} loss(-)={b10} net={b01 - b10:+d}")
    # per-class collapse -> typed
    if "typed" in arms and "collapse" in arms:
        print("\n=== per-class acc collapse -> typed ===")
        for c in CANDS:
            ids = [i for i in use if labels[i] == c]
            if not ids:
                continue
            a0 = sum(results.get((i, "collapse")) == c for i in ids) / len(ids)
            a1 = sum(results.get((i, "typed")) == c for i in ids) / len(ids)
            print(f"  {c:12s} n={len(ids):3d}  collapse={a0 * 100:5.1f}%  typed={a1 * 100:5.1f}%  Δ={(a1 - a0) * 100:+5.1f}")

    os.makedirs(a.out, exist_ok=True)
    outp = f"{a.out}/soaratac_{a.model}_n{a.n}.json"
    json.dump({"acc": acc, "model": a.model, "n": a.n, "arms": arms, "kc": a.kc, "kt": a.kt,
               "truth": {str(i): labels[i] for i in use},
               "preds": {f"{i}|{arm}": results[(i, arm)] for (i, arm) in results}}, open(outp, "w"))
    print("saved", outp, f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
