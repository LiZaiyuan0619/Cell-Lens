#!/usr/bin/env python3
"""聚合 results_cite/ 下所有 cite_*.json, 出一张跨模型总表 + McNemar + per-class l0->f3。"""
import json, glob, collections, sys
CANDS7=["Naive CD4+ T","Effector CD4+ T","Naive CD8+ T","Effector CD8+ T","B","NK","Monocytes"]

def mcnemar(base, oth):
    b01=sum(1 for x,y in zip(base,oth) if (not x) and y)
    b10=sum(1 for x,y in zip(base,oth) if x and (not y))
    return b01,b10

files=sorted(glob.glob("results_cite/cite_*.json"))
print(f"# F3/CITE 聚合 ({len(files)} runs)\n")
print(f"{'model/run':38s} {'n':>4} {'l0':>6} {'l0spec':>7} {'f3':>6} {'f3shuf':>7} {'f3noval':>8} | f3vs_l0spec(net)")
print("-"*120)
allrows=[]
for f in files:
    d=json.load(open(f))
    arms=d["arms"]; preds=d["preds"]; labels={int(k):v for k,v in d["labels"].items()}
    by=collections.defaultdict(dict)
    for key,p in preds.items():
        i,arm=key.split("|"); by[int(i)][arm]=p
    cells=list(labels)
    acc={a:sum(by[i].get(a)==labels[i] for i in cells)/len(cells) for a in arms}
    base_arm="l0spec" if "l0spec" in arms else "l0"
    base=[by[i].get(base_arm)==labels[i] for i in cells]
    net={}
    for a in arms:
        if a==base_arm: continue
        b01,b10=mcnemar(base,[by[i].get(a)==labels[i] for i in cells]); net[a]=b01-b10
    name=f.replace("results_cite/cite_","").replace(".json","")
    def g(a): return f"{acc[a]*100:5.1f}" if a in acc else "   -  "
    print(f"{name:38s} {len(cells):>4} {g('l0'):>6} {g('l0spec'):>7} {g('f3'):>6} {g('f3shuf'):>7} {g('f3noval'):>8} | f3 net={net.get('f3','?'):+}")
    allrows.append((name,acc,net,by,labels,cells,arms))

# per-class l0->f3 (取最大 n 的 7类 run)
sevens=[r for r in allrows if set(r[5][:1]) and r[1].get('f3') is not None and any(l in CANDS7 for l in r[4].values())]
if sevens:
    r=max(sevens,key=lambda r:len(r[5]))
    name,acc,net,by,labels,cells,arms=r
    print(f"\n## per-class l0->f3  [{name}, n={len(cells)}]")
    print(f"{'class':18s} {'n':>4} {'l0':>7} {'f3':>7} {'Δ':>7}")
    for c in CANDS7:
        ids=[i for i in cells if labels[i]==c]
        if not ids: continue
        a0=sum(by[i].get('l0')==c for i in ids)/len(ids); a3=sum(by[i].get('f3')==c for i in ids)/len(ids)
        print(f"{c:18s} {len(ids):>4} {a0*100:6.1f}% {a3*100:6.1f}% {(a3-a0)*100:+6.1f}")
