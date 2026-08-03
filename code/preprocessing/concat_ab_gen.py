"""Multi-cell serialization A/B: does EXPLICIT per-cell typed boundary (CellCode structure)
help batch CTA vs FLAT concatenation? SAME cells, two formats (paired). Extensibility claim.
Outputs concat_structured.jsonl + concat_flat.jsonl for cellpuzzles_run.py --data.
"""
import numpy as np, json, argparse
from scipy.sparse import csr_matrix

ap = argparse.ArgumentParser()
ap.add_argument("--npz", default="raw_data/tenx_pbmc_cite/tenx_labeled.npz")
ap.add_argument("--M", type=int, default=60)      # puzzles
ap.add_argument("--topk", type=int, default=40)   # genes/cell
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = np.random.RandomState(a.seed)

d = np.load(a.npz, allow_pickle=True)
X = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
genes = np.array([str(g) for g in d["genes"]]); labels = np.array([str(l) for l in d["labels7"]])
keep = labels != "Unassigned"; X, labels = X[keep], labels[keep]
types = sorted(set(labels))
ev_by_type = {t: np.where(labels == t)[0].tolist() for t in types}

def cell_top(i):
    row = X.getrow(i); order = row.indices[np.argsort(row.data)[::-1]]
    return [genes[j] for j in order[:a.topk]]

SYS_S = ("You are an expert in single-cell annotation. You are given a batch of N single cells, each "
         "delimited by a 'Cell k:' header, with its top expressed genes. Assign each of the N candidate "
         "cell types to exactly one cell, reasoning jointly over the batch. "
         "End with <answer>type | type | ...</answer> in cell order.")
SYS_F = ("You are an expert in single-cell annotation. Below are the top expressed genes of N single cells, "
         "concatenated in order and separated by ' ; '. Segment them into the N cells and assign each of the "
         "N candidate cell types to exactly one cell, reasoning jointly. "
         "End with <answer>type | type | ...</answer> in cell order.")

def gen():
    S, F = [], []
    for _ in range(a.M):
        ts = list(types); rng.shuffle(ts)
        cells = [(t, cell_top(int(rng.choice(ev_by_type[t])))) for t in ts]
        gold = " | ".join(t for t, _ in cells)
        # A: structured (explicit typed cell boundaries = CellCode)
        ls = ["Context: Human PBMC sample.", ""]
        for k, (t, gl) in enumerate(cells, 1):
            ls += [f"Cell {k}: {', '.join(gl)}", ""]
        ls += ["Match the cells above to one of the following cell types:"] + ts
        S.append({"system_msg": SYS_S, "user_msg": "\n".join(ls), "assistant_msg": gold})
        # B: flat (bare ';' runs, no per-cell headers → model must segment)
        runs = " ; ".join(" ".join(gl) for _, gl in cells)
        lf = ["Context: Human PBMC sample.", "",
              f"Top genes of {len(cells)} cells, in order, separated by ' ; ':", runs, "",
              "Assign each cell (in order) one of these cell types:"] + ts
        F.append({"system_msg": SYS_F, "user_msg": "\n".join(lf), "assistant_msg": gold})
    return S, F

S, F = gen()
print(f"{X.shape[0]} cells, N={len(types)} types/puzzle: {types}")
for tag, dat in [("structured", S), ("flat", F)]:
    fn = f"new_benchmarks/concat_{tag}.jsonl"
    with open(fn, "w") as f:
        for p in dat:
            f.write(json.dumps(p) + "\n")
    print(f"saved {fn} ({len(dat)} puzzles)")
