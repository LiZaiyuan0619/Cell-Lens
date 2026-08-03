#!/usr/bin/env python3
"""统计显著性: 对二分类 A/B harness 的结果 JSON (doublet/prolif/highmito 同构) 算
  - 配对 McNemar exact p (binomial, 针对 net 的不一致格)
  - net (=gain-loss) 的 bootstrap 95% CI
  - 每臂 acc 的 Wilson 95% CI
用法: .venv/bin/python significance.py results_doublet/doublet_deepseek_v4_flash_n150.json [more.json ...]
"""
import sys, json
import numpy as np
from scipy import stats

def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (max(0, c-h), min(1, c+h))

def analyze(path):
    o = json.load(open(path))
    arms = o["arms"]; truth = o["truth"]; preds = o["preds"]
    ks = sorted(truth.keys(), key=int); n = len(ks)
    print(f"\n### {path}  (model={o.get('model')}, n={n})")
    corr = {}
    for arm in arms:
        c = np.array([1 if preds.get(f"{k}|{arm}") == truth[k] else 0 for k in ks])
        corr[arm] = c
        lo, hi = wilson(c.sum(), n)
        print(f"  {arm:8s} acc={c.mean()*100:5.1f}%  (Wilson95 {lo*100:.1f}-{hi*100:.1f})")
    if "l0" not in arms: return
    base = corr["l0"]; rng = np.random.default_rng(0)
    for arm in arms:
        if arm == "l0": continue
        oth = corr[arm]
        b01 = int(((base == 0) & (oth == 1)).sum())   # l0 错→arm 对 (gain)
        b10 = int(((base == 1) & (oth == 0)).sum())   # l0 对→arm 错 (loss)
        net = b01 - b10
        m = b01 + b10
        p = stats.binomtest(min(b01, b10), m, 0.5).pvalue if m > 0 else 1.0  # McNemar exact (two-sided)
        diff = oth - base  # per-item net contribution in {-1,0,1}
        boot = np.array([diff[rng.integers(0, n, n)].sum() for _ in range(5000)])
        cl, ch = np.percentile(boot, [2.5, 97.5])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  l0 vs {arm:8s} net={net:+d} (gain={b01} loss={b10})  McNemar p={p:.4g} {sig}  "
              f"net bootstrap95=[{cl:+.0f},{ch:+.0f}]  Δacc={net/n*100:+.1f}pp")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
