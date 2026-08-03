#!/usr/bin/env python3
"""Create machine-readable summaries for every valid archived result file.

The script never calls a model.  It scans saved JSON outputs, records their
stored aggregate metrics, and recomputes exact-match statistics for the common
paired ``truth``/``preds`` schema.  Paired comparisons use two-sided exact
McNemar tests and a seed-0, 5,000-resample bootstrap confidence interval.

Usage
-----
python code/analysis/summarize_archived_results.py \
  --results results --out results/statistics
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def exact_correctness(obj: dict) -> tuple[list[str], dict[str, np.ndarray]] | None:
    truth = obj.get("truth") if obj.get("truth") is not None else obj.get("labels")
    preds, arms = obj.get("preds"), obj.get("arms")
    if not isinstance(preds, dict) or not isinstance(arms, list):
        return None
    if isinstance(truth, dict):
        keys = sorted(truth, key=lambda value: int(value) if str(value).isdigit() else str(value))
        gold = {str(key): truth[key] for key in keys}
    elif isinstance(truth, list):
        keys = [str(index) for index in range(len(truth))]
        gold = dict(zip(keys, truth))
    else:
        return None
    corr = {
        arm: np.asarray([preds.get(f"{key}|{arm}") == gold[key] for key in keys], dtype=np.int8)
        for arm in arms
    }
    return keys, corr


def preferred_baseline(arms: list[str]) -> str | None:
    for name in ("l0", "l0spec", "baseline", "collapse", "flat"):
        if name in arms:
            return name
    return arms[0] if arms else None


def stored_rows(relpath: str, obj: dict) -> list[dict[str, object]]:
    common = {"file": relpath, "model": obj.get("model", ""), "schema": "stored_aggregate"}
    rows: list[dict[str, object]] = []
    for metric in ("batch_acc", "cell_acc", "det_acc", "lineage_acc", "full_acc"):
        value = obj.get(metric)
        if isinstance(value, (int, float)):
            rows.append({**common, "arm": obj.get("arm", ""), "metric": metric,
                         "value": value, "n": obj.get("n", ""), "ci_low": "", "ci_high": ""})
    if isinstance(obj.get("exact"), (int, float)) and isinstance(obj.get("n"), int):
        k, n = int(obj["exact"]), int(obj["n"])
        lo, hi = wilson(k, n)
        rows.append({**common, "arm": obj.get("arm", ""), "metric": "exact_accuracy",
                     "value": k / n if n else "", "n": n, "ci_low": lo, "ci_high": hi})
    summary = obj.get("summary")
    if isinstance(summary, dict):
        for arm, values in summary.items():
            if not isinstance(values, dict):
                continue
            for metric, value in values.items():
                if isinstance(value, (int, float)):
                    rows.append({**common, "arm": arm, "metric": metric, "value": value,
                                 "n": obj.get("n", ""), "ci_low": "", "ci_high": ""})
    return rows


def analyze(path: Path, root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    relpath = path.relative_to(root.parent).as_posix()
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [], [], {"file": relpath, "status": "unreadable", "schema": "", "detail": str(exc)}
    if not isinstance(obj, dict):
        return [], [], {"file": relpath, "status": "unsupported", "schema": type(obj).__name__, "detail": "top level is not an object"}

    summaries = stored_rows(relpath, obj)
    comparisons: list[dict[str, object]] = []
    parsed = exact_correctness(obj)
    schema = "stored_aggregate"
    if parsed is not None:
        schema = "paired_exact"
        keys, corr = parsed
        n = len(keys)
        arms = list(corr)
        for arm, values in corr.items():
            k = int(values.sum())
            lo, hi = wilson(k, n)
            summaries.append({"file": relpath, "model": obj.get("model", ""), "schema": schema,
                              "arm": arm, "metric": "exact_accuracy", "value": k / n if n else "",
                              "n": n, "ci_low": lo, "ci_high": hi})
        baseline = preferred_baseline(arms)
        if baseline is not None:
            base = corr[baseline]
            for arm in arms:
                if arm == baseline:
                    continue
                other = corr[arm]
                gains = int(((base == 0) & (other == 1)).sum())
                losses = int(((base == 1) & (other == 0)).sum())
                discordant = gains + losses
                pvalue = stats.binomtest(gains, discordant, 0.5).pvalue if discordant else 1.0
                delta = other.astype(float) - base.astype(float)
                rng = np.random.default_rng(0)
                if n:
                    draws = rng.integers(0, n, size=(5000, n))
                    boot = delta[draws].mean(axis=1)
                    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
                else:
                    ci_low = ci_high = float("nan")
                comparisons.append({"file": relpath, "model": obj.get("model", ""),
                                    "baseline": baseline, "comparison": arm, "n": n,
                                    "gain_pairs": gains, "loss_pairs": losses,
                                    "delta_accuracy": float(delta.mean()) if n else "",
                                    "mcnemar_exact_p": pvalue,
                                    "bootstrap_delta_ci_low": ci_low,
                                    "bootstrap_delta_ci_high": ci_high,
                                    "bootstrap_resamples": 5000, "bootstrap_seed": 0})
    status = "summarized" if summaries else "valid_no_supported_metric"
    return summaries, comparisons, {"file": relpath, "status": status, "schema": schema, "detail": ""}


def write_tsv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("results/statistics"))
    args = parser.parse_args()
    summaries: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    manifest: list[dict[str, object]] = []
    output_root = args.out.resolve()
    paper_table_root = (args.results / "paper_tables").resolve()
    for path in sorted(args.results.rglob("*")):
        if (not path.is_file() or output_root in path.resolve().parents
                or paper_table_root in path.resolve().parents or path.name == "_DONE"
                or path.suffix not in ("", ".json")):
            continue
        srows, crows, mrow = analyze(path, args.results)
        summaries.extend(srows)
        comparisons.extend(crows)
        manifest.append(mrow)
    write_tsv(args.out / "result_metrics.tsv", summaries,
              ["file", "model", "schema", "arm", "metric", "value", "n", "ci_low", "ci_high"])
    write_tsv(args.out / "paired_comparisons.tsv", comparisons,
              ["file", "model", "baseline", "comparison", "n", "gain_pairs", "loss_pairs",
               "delta_accuracy", "mcnemar_exact_p", "bootstrap_delta_ci_low",
               "bootstrap_delta_ci_high", "bootstrap_resamples", "bootstrap_seed"])
    write_tsv(args.out / "result_file_manifest.tsv", manifest,
              ["file", "status", "schema", "detail"])
    print(f"summarized {len(manifest)} files; {len(summaries)} metrics; {len(comparisons)} paired comparisons")


if __name__ == "__main__":
    main()
