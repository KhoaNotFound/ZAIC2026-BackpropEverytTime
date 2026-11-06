"""Local deterministic scorer. Does not invoke an LLM or load model weights."""

import argparse
import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

from zaic.formats import number, read_jsonl, valid_answer


def normalize(text):
    text = unicodedata.normalize("NFC", text).casefold()
    return "".join(
        c
        for c in text
        if not c.isspace() and not unicodedata.category(c).startswith("P")
    )


def equal_value(pred, gold, aliases=()):
    if isinstance(gold, str):
        return isinstance(pred, str) and normalize(pred) in {
            normalize(s) for s in [gold, *aliases] if isinstance(s, str)
        }
    if type(gold) is bool:
        return type(pred) is bool and pred == gold
    if number(gold):
        # Tight absolute tolerance only: no relative tolerance that accepts VND errors on large sums.
        return number(pred) and abs(Decimal(str(pred)) - Decimal(str(gold))) <= Decimal(
            "1e-8"
        )
    return False


def aliases_at(aliases, index):
    if index >= len(aliases):
        return []
    value = aliases[index]
    return value if isinstance(value, list) else [value]


def correct(pred, gold):
    if pred.get("type") != gold["type"] or not valid_answer(pred):
        return False
    kind = gold["type"]
    p, g, a = pred.get("answer"), gold.get("answer"), gold.get("alias", [])
    if kind == "unanswerable":
        return True  # Type is scored; reason is required, not judged by another LLM.
    if kind == "list":
        return len(p) == len(g) and all(
            equal_value(x, y, aliases_at(a, i)) for i, (x, y) in enumerate(zip(p, g))
        )
    if kind == "table":
        pc, gc = p["columns"], g["columns"]
        if len(pc) != len(gc) or len(p["rows"]) != len(g["rows"]):
            return False
        pnorm, gnorm = [normalize(c) for c in pc], [normalize(c) for c in gc]
        if len(set(pnorm)) != len(pnorm):
            return False
        # Match by names when possible, otherwise use positional columns per rulebook.
        indices = [pnorm.index(c) if c in pnorm else i for i, c in enumerate(gnorm)]
        if len(set(indices)) != len(indices):
            return False
        candidates = []
        for grow_index, grow in enumerate(g["rows"]):
            matches = []
            for i, prow in enumerate(p["rows"]):
                ordered = [prow[j] for j in indices]
                if all(
                    equal_value(x, y, aliases_at(a, grow_index) if col == 0 else ())
                    for col, (x, y) in enumerate(zip(ordered, grow))
                ):
                    matches.append(i)
            candidates.append(matches)
        # Bipartite matching handles repeated names/aliases without reusing predicted rows.
        assigned = {}

        def match(gidx, visited):
            for pidx in candidates[gidx]:
                if pidx in visited:
                    continue
                visited.add(pidx)
                if pidx not in assigned or match(assigned[pidx], visited):
                    assigned[pidx] = gidx
                    return True
            return False

        return all(match(i, set()) for i in range(len(candidates)))
    return equal_value(p, g, a if kind == "string" else ())


def score(predictions, answers, ids=None, questions=None):
    golds = {a["id"]: a for a in answers}
    preds = {p["id"]: p for p in predictions}
    ids = list(golds) if ids is None else ids
    if len(ids) != len(set(ids)) or set(ids) - set(golds):
        raise ValueError("Evaluation IDs must be unique and present in answers")
    if set(preds) - set(golds):
        raise ValueError("Predictions contain IDs absent from answers")
    qmap = {q["id"]: q for q in questions or []}
    counts, details = Counter(), []
    by_type, by_business = defaultdict(Counter), defaultdict(Counter)
    for ident in ids:
        gold, pred = golds[ident], preds.get(ident)
        if pred is None or pred.get("type") == "abstain":
            outcome = "abstained"
        else:
            outcome = "correct" if correct(pred, gold) else "incorrect"
        counts[outcome] += 1
        by_type[gold["type"]][outcome] += 1
        if ident in qmap:
            by_business[qmap[ident]["business_id"]][outcome] += 1
        details.append(
            {
                "id": ident,
                "question": qmap.get(ident, {}).get("question"),
                "outcome": outcome,
                "points": {"correct": 1, "incorrect": -1, "abstained": 0}[outcome],
                "expected_type": gold["type"],
                "prediction": pred,
                "expected": gold,
            }
        )

    def summarize(c):
        n = sum(c.values())
        attempted = c["correct"] + c["incorrect"]
        return {
            "total": n,
            **{key: c[key] for key in ["correct", "incorrect", "abstained"]},
            "accuracy": c["correct"] / n if n else 0,
            "coverage": attempted / n if n else 0,
            "attempted_accuracy": c["correct"] / attempted if attempted else 0,
            "score": c["correct"] - c["incorrect"],
            "normalized_score": (c["correct"] - c["incorrect"]) / n if n else 0,
        }

    return {
        "overall": summarize(counts),
        "by_type": {k: summarize(v) for k, v in by_type.items()},
        "by_business": {k: summarize(v) for k, v in by_business.items()},
        "details": details,
        "note": "Local scorer follows published rulebook and supplied aliases; not an official scorer. "
        "Additional valid source-name variants absent from aliases may require manual review.",
    }


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": values[min(len(values) - 1, int(0.9 * len(values)))],
        "max": values[-1],
        "total": sum(values),
    }


def runtime(metrics, details, manifest=None):
    """Join per-question run metrics with scoring outcomes."""
    by_id = {m["id"]: m for m in metrics}
    outcome_by_status = defaultdict(Counter)
    for d in details:
        status = by_id.get(d["id"], {}).get("status", "not_run")
        d["harness_status"] = status
        outcome_by_status[status][d["outcome"]] += 1
    peak = defaultdict(lambda: {"allocated_peak_gib": 0.0, "reserved_peak_gib": 0.0})
    for m in metrics:
        for gpu, mem in (m.get("gpu_memory") or {}).items():
            for key in peak[gpu]:
                peak[gpu][key] = max(peak[gpu][key], mem.get(key, 0.0))
    config = (manifest or {}).get("config", {})
    return {
        "model": config.get("model"),
        "quantization": config.get("quantization"),
        "device_map_mode": config.get("device_map"),
        "load_seconds": (manifest or {}).get("load_seconds"),
        "questions_run": len(metrics),
        # ok = model gave a final answer; stalled/max_steps/error are harness-level stops.
        "status": dict(Counter(m["status"] for m in metrics)),
        "outcome_by_status": {k: dict(v) for k, v in outcome_by_status.items()},
        "seconds_per_question": stats(m.get("elapsed_seconds") for m in metrics),
        "steps_per_question": stats(m.get("steps") for m in metrics),
        "tool_calls_per_question": stats(m.get("tool_calls") for m in metrics),
        "input_tokens_per_question": stats(m.get("input_tokens") for m in metrics),
        "output_tokens_per_question": stats(m.get("output_tokens") for m in metrics),
        "peak_gpu_memory_gib": dict(peak),
    }


def print_report(report):
    o = report["overall"]
    print(
        f"Overall: {o['correct']}/{o['total']} correct, {o['incorrect']} incorrect, "
        f"{o['abstained']} abstained | accuracy {o['accuracy']:.1%}, "
        f"coverage {o['coverage']:.1%}, attempted accuracy {o['attempted_accuracy']:.1%}, "
        f"score {o['score']}"
    )
    print("By answer type:")
    for kind, s in sorted(report["by_type"].items()):
        print(
            f"  {kind:<13} {s['correct']:>3}/{s['total']:<3} correct  "
            f"{s['incorrect']:>3} incorrect  {s['abstained']:>3} abstained"
        )
    print("Per-question comparison (prediction vs expected):")
    for detail in report["details"]:
        print(json.dumps(detail, ensure_ascii=False))
    r = report.get("runtime")
    if not r:
        return
    load = r["load_seconds"]
    print(
        f"Model: {r['model']} ({r['quantization']}, device map {r['device_map_mode']}), "
        f"load {load:.0f}s" if load is not None else f"Model: {r['model']}"
    )
    print("Run status:", r["status"])
    print("Outcome by status:", r["outcome_by_status"])
    for key in ["seconds_per_question", "steps_per_question", "tool_calls_per_question",
                "input_tokens_per_question", "output_tokens_per_question"]:
        s = r[key]
        if s:
            print(
                f"  {key:<27} mean {s['mean']:.1f}  median {s['median']:.1f}  "
                f"p90 {s['p90']:.1f}  max {s['max']:.1f}"
            )
    for gpu, mem in sorted(r["peak_gpu_memory_gib"].items()):
        print(
            f"  GPU {gpu} peak: {mem['allocated_peak_gib']:.2f} GiB allocated, "
            f"{mem['reserved_peak_gib']:.2f} GiB reserved"
        )


def evaluate_run(predictions_path, answers_path, questions_path=None, scope="run", output=None):
    """Score saved predictions against labels only after inference has finished."""
    preds, golds = read_jsonl(predictions_path), read_jsonl(answers_path)
    manifest_path = predictions_path.with_suffix(".run.json")
    if scope == "run":
        if not manifest_path.exists():
            raise SystemExit(
                "Missing .run.json manifest. Use --scope predictions or --scope all explicitly."
            )
        ids = json.loads(manifest_path.read_text())["config"]["selected_ids"]
    else:
        ids = [v["id"] for v in preds] if scope == "predictions" else None
    report = score(
        preds, golds, ids, read_jsonl(questions_path) if questions_path else None
    )
    metrics_path = predictions_path.with_suffix(".metrics.jsonl")
    if metrics_path.exists():
        manifest = (
            json.loads(manifest_path.read_text()) if manifest_path.exists() else None
        )
        report["runtime"] = runtime(
            read_jsonl(metrics_path), report["details"], manifest
        )
    output = output or predictions_path.with_suffix(".evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print_report(report)
    print(f"Details: {output}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--questions", type=Path, help="Optional: group scores by business")
    p.add_argument(
        "--scope",
        choices=["run", "predictions", "all"],
        default="run",
        help="run uses inference manifest (includes missing outputs); predictions scores only existing output IDs; all scores all gold questions",
    )
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    evaluate_run(args.predictions, args.answers, args.questions, args.scope, args.output)


if __name__ == "__main__":
    main()
