#!/usr/bin/env python3
"""表示>模型 收敛分析: 跨模型 panel, L0(raw cell-sentence) vs f3(恢复模态) 的准确率分散度。
皇冠明珠主图: L0 各模型分散 -> 恢复后收敛; 且弱模型(低L0)涨更多(gain~L0 负相关)。
用法: python analyze_convergence.py
"""
import json, glob, os, statistics as st

def load_acc(path, arm):
    d = json.load(open(path)); preds=d['preds']; labels=d['labels']
    cells=sorted(set(k.split('|')[0] for k in preds))
    vals=[int(preds.get(f"{c}|{arm}")==labels.get(c)) for c in cells if f"{c}|{arm}" in preds]
    return 100.0*sum(vals)/len(vals) if vals else None

NAME={'gpt54_mini':'gpt-mini','deepseek_v4_flash':'deepseek-flash','gemini35_flash':'gemini',
      'deepseek_v4_pro':'deepseek-pro','qwen37_plus':'qwen-plus','qwen37_max':'qwen-max',
      'qwen36_flash':'qwen3.6-flash','qwen35_397b':'qwen3.5-397b'}

# canonical 10 模型 sweep(results_conv_*, 统一 native_fmt+kg100); 不存在则回退旧 struct/struct2(6 模型)
if os.path.isdir('results_conv_10x') and glob.glob('results_conv_10x/cite_*.json'):
    SRC={'10x':['results_conv_10x'],'mimitou':['results_conv_mimitou'],'hao':['results_conv_hao']}
else:
    SRC={'10x':['results_struct_10x','results_struct2_10x'],
         'mimitou':['results_struct_mimitou','results_struct2_mimitou']}

def corr(xs,ys):
    n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
    num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx=sum((x-mx)**2 for x in xs)**.5; dy=sum((y-my)**2 for y in ys)**.5
    return num/(dx*dy) if dx*dy else 0.0

for src,dirs in SRC.items():
    rows=[]; seen=set()
    for d in dirs:
        for f in sorted(glob.glob(f'{d}/cite_*.json')):
            mid=os.path.basename(f).replace('cite_','').replace('_n50_7.json','')
            nm=NAME.get(mid,mid)
            if 'struct2' in d and mid=='gpt54_mini': nm='gpt-mini(HIGH)'
            if nm in seen: continue
            seen.add(nm)
            l0=load_acc(f,'l0spec'); f3=load_acc(f,'f3')
            if l0 is not None and f3 is not None: rows.append((nm,l0,f3,f3-l0))
    print(f"\n===== source: {src}  ({len(rows)} models) =====")
    print(f"{'model':16} {'L0':>6} {'f3':>6} {'gain':>6}")
    for nm,l0,f3,g in sorted(rows,key=lambda r:r[1]):
        print(f"{nm:16} {l0:6.1f} {f3:6.1f} {g:+6.1f}")
    L0s=[r[1] for r in rows]; F3s=[r[2] for r in rows]; Gs=[r[3] for r in rows]
    print(f"  --- 收敛统计 ---")
    print(f"  L0 范围 {min(L0s):.1f}-{max(L0s):.1f} (spread {max(L0s)-min(L0s):.1f}pp, std {st.pstdev(L0s):.1f})")
    print(f"  f3 范围 {min(F3s):.1f}-{max(F3s):.1f} (spread {max(F3s)-min(F3s):.1f}pp, std {st.pstdev(F3s):.1f})")
    print(f"  → std 收窄 {st.pstdev(L0s):.1f} → {st.pstdev(F3s):.1f} ({'收敛✓' if st.pstdev(F3s)<st.pstdev(L0s) else '未收窄✗'})")
    print(f"  gain~L0 相关 r={corr(L0s,Gs):+.2f} ({'弱模型涨更多✓=收敛机制' if corr(L0s,Gs)<-0.3 else '不明显'})")
