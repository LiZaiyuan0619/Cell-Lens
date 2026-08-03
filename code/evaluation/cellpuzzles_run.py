"""Cell-o1 / CellPuzzles 第二 benchmark runner (batch-level 唯一分配谜题, 确定性评分)。
官方格式: system_msg+user_msg → 模型输出 <answer>type1 | type2 | ...</answer>; gold=assistant_msg(' | '分隔)。
评分: cell-level acc(按位) + batch-level acc(整批全对)。
用法: python cellpuzzles_run.py --model deepseek-v4-flash --n 20 --out results_cellpuzzles/ds_L0.json
"""
import json, os, re, time, argparse, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_URL = "https://api.example.com/v1"
# 3 个独立可叠加端点 (分散负载提吞吐, 见 API endpoint rotation-throughput 记忆)
EPS = {"endpoint1": "https://api.example.com/v1", "endpoint2": "https://api.example.com/v1",
       "endpoint3": "https://api.example.com/v1"}
DATA = os.path.join(PACKAGE_ROOT, "data", "public_benchmark_inputs", "cellpuzzles_test.jsonl")


def load_key():
    return os.environ.get("LLM_API_KEY")


def extract_answer(text):
    m = re.search(r"<answer>(.*?)</answer>", text, re.S | re.I)
    seg = m.group(1) if m else text.split("</think>")[-1]
    # take the line with ' | ' if multiple
    cand = [ln for ln in seg.strip().splitlines() if "|" in ln]
    seg = cand[-1] if cand else seg
    return [x.strip() for x in seg.strip().strip("|").split("|") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--endpoints", default="hk,global,endpoint3", help="逗号分隔, 轮询分散负载提吞吐")
    ap.add_argument("--max_tokens", type=int, default=16384)  # 推理模型需足够 token (think+answer)，4000 会被think耗光→空content
    ap.add_argument("--base_url", default=BASE_URL)
    ap.add_argument("--data", default=DATA, help="jsonl path; 用增强版(rneg)时指向新文件")
    ap.add_argument("--reasoning_effort", default="", help="gpt-5.4-mini: low/medium/high/xhigh(默认none不思考)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # --- resolve model via models.yaml if --model is a yaml alias (alt flagships etc.) ---
    # If --model is a raw model string not in YAML, use the configured API endpoint.
    import yaml as _yaml
    _cfg = _yaml.safe_load(open(os.path.join(PACKAGE_ROOT, "AtomWorldBench/src/config/models.yaml")))
    model_name = args.model
    if args.model in _cfg:
        y = _cfg[args.model]
        model_name = y["model_name"]
        key = os.environ.get(y["api_key"].strip("${}")) or load_key()
        # a yaml-resolved model pins its own single endpoint (ignore the Endpoint 3 EPS rotation)
        clients = [OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or y["base_url"], timeout=600, max_retries=1)]
    else:
        key = load_key()
        clients = [OpenAI(api_key=key, base_url=os.environ.get("LLM_BASE_URL") or EPS.get(e.strip(), e.strip()), timeout=600, max_retries=1)
                   for e in args.endpoints.split(",") if e.strip()]
    data = [json.loads(l) for l in open(args.data)][: args.n]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def call(pair):
        idx, ex = pair
        msgs = [{"role": "system", "content": ex["system_msg"]},
                {"role": "user", "content": ex["user_msg"]}]
        client = clients[idx % len(clients)]   # 轮询端点分散负载
        for attempt in range(4):
            try:
                kw = dict(model=model_name, messages=msgs, temperature=1.0,
                          max_tokens=args.max_tokens, timeout=600)
                if args.reasoning_effort:
                    kw["reasoning_effort"] = args.reasoning_effort
                r = client.chat.completions.create(**kw)
                return idx, (r.choices[0].message.content or "")
            except Exception as e:
                last = f"{type(e).__name__}:{str(e)[:80]}"
                time.sleep(2 * (attempt + 1))
        return idx, f"[[ERROR {last}]]"

    out = [None] * len(data)
    done = 0; lock = threading.Lock(); t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for fut in as_completed([ex.submit(call, (i, d)) for i, d in enumerate(data)]):
            idx, resp = fut.result(); out[idx] = resp
            with lock:
                done += 1
                if done % 20 == 0 or done == len(data):
                    print(f"{done}/{len(data)} [{time.time()-t0:.0f}s]", flush=True)

    # score: cell-level (per-position) + batch-level (whole batch exact)
    cell_correct = cell_total = batch_correct = errs = 0
    rows = []
    for ex, resp in zip(data, out):
        gold = [x.strip() for x in ex["assistant_msg"].strip().strip("|").split("|") if x.strip()]
        if resp.startswith("[[ERROR"): errs += 1
        pred = extract_answer(resp)
        # align by position; if length mismatch, pad
        n = len(gold)
        cc = sum(1 for i in range(n) if i < len(pred) and pred[i].lower() == gold[i].lower())
        cell_correct += cc; cell_total += n
        if cc == n and len(pred) == n: batch_correct += 1
        rows.append({"gold": gold, "pred": pred, "cell_correct": cc, "n": n})
    N = len(data)
    res = {"model": args.model, "n": N,
           "cell_acc": 100.0 * cell_correct / cell_total if cell_total else 0,
           "batch_acc": 100.0 * batch_correct / N if N else 0,
           "errs": errs}
    print(f"CellPuzzles [{args.model}]: cell_acc={res['cell_acc']:.1f}%  "
          f"batch_acc={res['batch_acc']:.1f}%  (N={N} puzzles, errs={errs})", flush=True)
    json.dump({**res, "rows": rows, "answers": out}, open(args.out, "w"), indent=1)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
