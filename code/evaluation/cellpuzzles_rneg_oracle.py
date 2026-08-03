"""CellPuzzles rneg 可行性 kill-test: 用 test 集 gold 反挖 oracle 标记 (数据自身符号, 必 in-vocab)。
*** 有泄漏的上界 ***: 只回答"正确 in-vocab 标记能否帮上 CellPuzzles"。帮->投入拿干净标记; 不帮->放弃。
gold 对齐: runner 按位打分(pred[i] vs gold[i]) => assistant_msg 第 i 个 type 对应 Cell i+1。
用法: python cellpuzzles_rneg_oracle.py --max_markers 12 --topk_pool 40 --out Cell-o1_repo/cellpuzzles_test_oracle.jsonl
"""
import json, os, re, argparse, collections

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "Cell-o1_repo", "cellpuzzles_test.jsonl")
# 常见高表达管家/线粒体/核糖体基因, 无判别力, 从 oracle 标记里剔除
STOP = set()
STOP_PREFIX = ("MT-", "MTND", "MTCO", "MTATP", "MTCYB", "RPL", "RPS", "RP11-", "RP5-", "RP4-", "AC0", "AL0", "LINC")
STOP_EXACT = {"ACTB", "ACTG1", "MALAT1", "TMSB4X", "TMSB10", "B2M", "EEF1A1", "FTL", "FTH1",
              "TPT1", "PTMA", "VIM", "NEAT1", "RPLP0", "RPLP1", "RPLP2", "GAPDH", "UBB", "UBC",
              "FAU", "TffMSB", "HLA-A", "HLA-B", "HLA-C", "MT2A", "MT1X"}


def is_stop(g):
    if g in STOP_EXACT:
        return True
    return any(g.startswith(p) for p in STOP_PREFIX)


def parse_puzzle(um):
    m = re.search(r"Match the cells above to one of the following cell types:\s*(.+)$", um, re.S)
    candidates = [c.strip() for c in m.group(1).strip().splitlines() if c.strip()] if m else []
    body = um[:m.start()] if m else um
    cells = {}
    for cm in re.finditer(r"Cell (\d+):\s*([^\n]+)", body):
        cells[int(cm.group(1))] = [g.strip() for g in cm.group(2).replace(",", " ").split() if g.strip()]
    ctx_m = re.search(r"Cell 1:", body)
    context = body[:ctx_m.start()].strip() if ctx_m else ""
    return context, cells, candidates


def mine_oracle(data, max_markers, topk_pool):
    """type -> oracle markers: 该类型细胞 top-pool 基因里出现频率最高、且非管家、且非全局高频。"""
    type_gene = collections.defaultdict(collections.Counter)
    type_n = collections.Counter()
    global_gene = collections.Counter()
    for ex in data:
        _, cells, _ = parse_puzzle(ex["user_msg"])
        gold = [t.strip() for t in ex["assistant_msg"].strip().strip("|").split("|") if t.strip()]
        for i, idx in enumerate(sorted(cells)):
            if i >= len(gold):
                break
            t = gold[i]
            type_n[t] += 1
            for g in cells[idx][:topk_pool]:
                if not is_stop(g):
                    type_gene[t][g] += 1
                    global_gene[g] += 1
    n_types_total = sum(type_n.values())
    name2mk = {}
    for t, cnt in type_gene.items():
        # score = 该类型内频率 - 全局基线频率 (富集), 取正富集 top
        scored = []
        for g, c in cnt.items():
            in_rate = c / type_n[t]
            base_rate = global_gene[g] / n_types_total
            scored.append((in_rate - base_rate, g))
        scored.sort(reverse=True)
        name2mk[t] = [g for s, g in scored[:max_markers] if s > 0]
    return name2mk, type_n


def make_rneg_um(um, name2mk, mode="both"):
    """mode: both=present+absent; present=只标 present(knowledge-injection); absent=只标 absent(纯恢复负证据)。"""
    context, cells, candidates = parse_puzzle(um)
    cand_mk = {c: name2mk.get(c, []) for c in candidates}
    desc = {"both": "'present' = marker is highly expressed in this cell; 'absent' = marker is NOT in the top genes (negative evidence against that type).",
            "present": "'present' = marker of this type is highly expressed in this cell.",
            "absent": "'absent' = marker of this type is NOT in the cell's top genes (negative evidence to rule the type out)."}[mode]
    lines = [context, ""]
    lines.append("For each cell, marker genes enriched in each candidate cell type are checked against "
                 "the cell's top expressed genes. " + desc)
    lines.append("")
    for idx in sorted(cells):
        gset = set(cells[idx])
        lines.append(f"Cell {idx}: {', '.join(cells[idx])}")
        ev = []
        for c in candidates:
            mks = cand_mk[c]
            if not mks:
                continue
            present = [m for m in mks if m in gset]
            absent = [m for m in mks if m not in gset]
            if mode == "present":
                ev.append(f"  - {c}: present[{', '.join(present) if present else 'none'}]")
            elif mode == "absent":
                ev.append(f"  - {c}: absent[{', '.join(absent) if absent else 'none'}]")
            else:
                ev.append(f"  - {c}: present[{', '.join(present) if present else 'none'}] "
                          f"absent[{', '.join(absent) if absent else 'none'}]")
        if ev:
            lines.append("  Marker evidence:")
            lines.extend(ev)
        lines.append("")
    lines.append("Match the cells above to one of the following cell types:")
    lines.extend(candidates)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_markers", type=int, default=12)
    ap.add_argument("--topk_pool", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(HERE, "Cell-o1_repo", "cellpuzzles_test_oracle.jsonl"))
    ap.add_argument("--show", type=int, default=0)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(SRC)]
    name2mk, type_n = mine_oracle(data, args.max_markers, args.topk_pool)
    covered = sum(1 for t in name2mk if name2mk[t])
    print(f"oracle markers: {covered}/{len(name2mk)} types have markers", flush=True)
    # 抽查几个常见类型
    for t in ["smooth muscle cell", "classical monocyte", "B cell", "natural killer cell", "macrophage"]:
        if t in name2mk:
            print(f"  {t}: {name2mk[t][:8]}")
    with open(args.out, "w") as f:
        for i, ex in enumerate(data):
            f.write(json.dumps({**ex, "user_msg": make_rneg_um(ex["user_msg"], name2mk)}) + "\n")
            if i == args.show:
                print("\n===== sample #%d (前1800字) =====" % i)
                print(make_rneg_um(ex["user_msg"], name2mk)[:1800])
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
