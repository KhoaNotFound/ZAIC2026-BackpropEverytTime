"""Run a small, reproducible DataMate sample or one user question on Kaggle."""

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from zaic.formats import read_jsonl


def select_questions(questions, limit, seed, sampling):
    rng = random.Random(seed)
    if sampling == "first":
        result = questions[:]
    elif sampling == "random":
        result = questions[:]
        rng.shuffle(result)
    else:
        groups = defaultdict(list)
        for q in questions:
            groups[q["business_id"]].append(q)
        keys = sorted(groups)
        rng.shuffle(keys)
        for group in groups.values():
            rng.shuffle(group)
        result = []
        while any(groups.values()):
            for key in keys:
                if groups[key]:
                    result.append(groups[key].pop())
    return result if limit == 0 else result[:limit]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Split directory containing questions.jsonl, rulebook.md and data/",
    )
    p.add_argument("--model", default="Qwen/Qwen3.5-9B")
    p.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    p.add_argument(
        "--device-map",
        default="split",
        help="split: explicit decoder-layer partition across >=2 GPUs; auto/balanced: library placement; cuda:0: one GPU",
    )
    p.add_argument("--gpu-memory-gib", type=int, default=13)
    p.add_argument("--context-tokens", type=int, default=16384)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--limit", type=int, default=20, help="0 = all questions")
    p.add_argument(
        "--sampling",
        choices=["stratified", "random", "first"],
        default="stratified",
        help="stratified = round robin over shuffled businesses",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--business-id", help="Filter a business, required with --question")
    p.add_argument("--question", help="Ask one custom question")
    p.add_argument("--as-of-date", help="Required with --question: YYYY-MM-DD")
    p.add_argument("--output", type=Path, default=Path("outputs/predictions.jsonl"))
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/select source inputs without loading the model",
    )
    return p


def main():
    args = parser().parse_args()
    if (
        min(
            args.limit,
            args.max_steps - 1,
            args.context_tokens - 1,
            args.max_new_tokens - 1,
        )
        < 0
    ):
        raise SystemExit("Invalid negative limit or nonpositive step/token budget")
    if args.question:
        if not args.business_id or not args.as_of_date:
            raise SystemExit("--question requires --business-id and --as-of-date")
        import datetime

        datetime.date.fromisoformat(args.as_of_date)
        payload = [args.business_id, args.as_of_date, args.question]
        ident = (
            "custom_" + hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]
        )
        questions = [
            {
                "id": ident,
                "business_id": args.business_id,
                "as_of_date": args.as_of_date,
                "question": args.question,
            }
        ]
    else:
        questions = read_jsonl(args.data_dir / "questions.jsonl")
        if args.business_id:
            questions = [q for q in questions if q["business_id"] == args.business_id]
        questions = select_questions(questions, args.limit, args.seed, args.sampling)
    if not questions:
        raise SystemExit("No questions selected")
    from zaic.tools import Tools

    for q in questions:
        tools = Tools(args.data_dir, q["business_id"])
        for path in ("README.md", "rulebook.md"):
            tools.resolve(path)
        if not all(
            isinstance(q.get(k), str) and q[k] for k in ("question", "as_of_date")
        ):
            raise SystemExit(f"Invalid question: {q['id']}")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "selected": len(questions),
                    "businesses": sorted({q["business_id"] for q in questions}),
                    "ids": [q["id"] for q in questions],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta_path = args.output.with_suffix(".run.json")
    trace_path = args.output.with_suffix(".traces.jsonl")
    metrics_path = args.output.with_suffix(".metrics.jsonl")
    config = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in vars(args).items()
        if k not in {"resume", "dry_run", "limit"}
    }
    config["selected_ids"] = [q["id"] for q in questions]
    # Include source file hashes to prevent resuming against changed questions/rules.
    config["source_hashes"] = {
        name: hashlib.sha256((args.data_dir / name).read_bytes()).hexdigest()
        for name in ("rulebook.md", "questions.jsonl")
        if (args.data_dir / name).exists()
    }
    existing = []
    if args.resume and args.output.exists():
        if (
            not meta_path.exists()
            or json.loads(meta_path.read_text())["config"] != config
        ):
            raise SystemExit(
                "Resume configuration/sample changed. Use a new output path."
            )
        existing = read_jsonl(args.output)
    elif any(p.exists() for p in [args.output, meta_path, trace_path, metrics_path]):
        raise SystemExit("Output exists. Use --resume or a new --output path.")
    done = {r["id"] for r in existing}
    pending = [q for q in questions if q["id"] not in done]
    if not pending:
        print("All selected questions already completed.")
        return
    from zaic.agent import answer_question
    from zaic.model import QwenBackend

    load_start = time.perf_counter()
    backend = QwenBackend(
        args.model,
        args.quantization,
        args.device_map,
        args.context_tokens,
        args.max_new_tokens,
        args.thinking,
        args.gpu_memory_gib,
    )
    import importlib.metadata

    versions = {
        pkg: importlib.metadata.version(pkg)
        for pkg in [
            "torch",
            "transformers",
            "accelerate",
            "bitsandbytes",
            "pandas",
            "duckdb",
            "openpyxl",
        ]
    }
    meta = {
        "config": config,
        "versions": versions,
        "load_seconds": time.perf_counter() - load_start,
        "device_map": backend.device_map,
        "note": "Inference only; no training loss or weight updates.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    times, errors = [], 0
    with (
        args.output.open("a", encoding="utf-8") as out,
        trace_path.open("a", encoding="utf-8") as traces,
        metrics_path.open("a", encoding="utf-8") as metrics,
    ):
        for i, q in enumerate(pending, 1):
            backend.reset_peak()
            prediction, detail = answer_question(
                backend, args.data_dir, q, max_steps=args.max_steps
            )
            detail["gpu_memory"] = backend.memory()
            times.append(detail["elapsed_seconds"])
            errors += detail["status"] != "ok"
            traces.write(json.dumps(detail, ensure_ascii=False) + "\n")
            traces.flush()
            brief = {k: v for k, v in detail.items() if k != "trace"}
            metrics.write(json.dumps(brief, ensure_ascii=False) + "\n")
            metrics.flush()
            # Predictions are the commit marker for resume; flush them last.
            out.write(
                json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n"
            )
            out.flush()
            print(
                f"[{i}/{len(pending)}] {q['id']} {detail['status']} {prediction['type']} "
                f"{detail['elapsed_seconds']:.1f}s; remaining estimate "
                f"{statistics.mean(times) * (len(pending) - i) / 60:.1f}min",
                flush=True,
            )
            if args.question:
                print(json.dumps(prediction, ensure_ascii=False, indent=2))
    summary = {
        "completed_this_run": len(times),
        "execution_failures": errors,
        "mean_seconds": statistics.mean(times),
        "median_seconds": statistics.median(times),
        "estimated_3000_questions_hours": statistics.mean(times) * 3000 / 3600,
        "estimate_note": "Rough estimate from this sample; excludes load time and varies by business/tool complexity.",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
