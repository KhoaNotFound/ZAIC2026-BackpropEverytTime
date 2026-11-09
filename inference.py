"""Run a small, reproducible DataMate sample or one user question on Kaggle."""

import argparse
import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from zaic.formats import read_jsonl

# Must be set before torch initializes CUDA: reduces fragmentation OOM on 15 GiB T4s.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def question_seed(seed, question_id):
    """Stable per-question sampling seed, independent of order and of resumes."""
    digest = hashlib.sha256(f"{seed}:{question_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def keep_complete_lines(path, keep=None):
    """Rewrite a JSONL file without a torn final line and, if keep is given,
    without records whose id is not in keep. Returns the kept records."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut off when the process was killed
        if keep is None or record.get("id") in keep:
            records.append(record)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    return records


def repair_for_resume(output, trace_path, metrics_path):
    """Make an interrupted run resumable.

    Predictions are written last per question, so they decide what is done. A
    question killed after its trace/metrics line but before its prediction is
    dropped from those files and rerun, instead of being recorded twice.
    """
    existing = keep_complete_lines(output)
    done = {r["id"] for r in existing}
    if len(done) != len(existing):
        raise SystemExit(f"{output} has duplicate ids; use a new output path")
    for path in (trace_path, metrics_path):
        keep_complete_lines(path, done)
    return existing


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
    p.add_argument(
        "--model",
        default="Qwen/Qwen3.5-9B",
        help="Any Hugging Face causal LM with a chat template (local path or hub id)",
    )
    p.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    p.add_argument(
        "--device-map",
        default="split",
        help="split: explicit decoder-layer partition across >=2 GPUs; auto/balanced: library placement; cuda:0: one GPU",
    )
    p.add_argument("--gpu-memory-gib", type=int, default=13)
    p.add_argument("--context-tokens", type=int, default=32768)
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Valid actions in the h4 run had p99 774 tokens; longer replies were runaway loops.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Defaults are the Qwen non-thinking preset 0.7/0.8/20; use the model card's "
        "recommended values for other models. 0 = greedy (prone to repetition loops)",
    )
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--repetition-penalty", type=float, default=1.0)
    p.add_argument("--thinking", action="store_true")
    p.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=512,
        help="Process prompt in chunks to reduce peak VRAM; 0 disables chunking",
    )
    p.add_argument(
        "--no-prefix-cache",
        dest="prefix_cache",
        action="store_false",
        help="Re-prefill the whole conversation every step (slower; for debugging)",
    )
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
        "--max-hours",
        type=float,
        default=0,
        help="Stop cleanly before starting a question after this many hours (0 = no limit); "
        "continue later with --resume. Use it to stay under Kaggle's 12-hour commit limit.",
    )
    p.add_argument(
        "--evaluate", action="store_true",
        help="After inference, compare saved predictions with this split's answers.jsonl",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate/select source inputs without loading the model",
    )
    return p


def main():
    args = parser().parse_args()
    if args.evaluate and (args.question or not (args.data_dir / "answers.jsonl").is_file()):
        raise SystemExit("--evaluate requires dataset questions and answers.jsonl")
    if (
        min(
            args.limit,
            args.max_steps - 1,
            args.context_tokens - 1,
            args.max_new_tokens - 1,
            args.prefill_chunk_size,
            args.temperature,
            args.top_k,
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
        if k not in {"resume", "dry_run", "limit", "evaluate", "max_hours"}
    }
    from zaic.agent import HARNESS_VERSION

    config["harness_version"] = HARNESS_VERSION
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
        existing = repair_for_resume(args.output, trace_path, metrics_path)
    elif any(p.exists() for p in [args.output, meta_path, trace_path, metrics_path]):
        raise SystemExit("Output exists. Use --resume or a new --output path.")
    done = {r["id"] for r in existing}
    pending = [q for q in questions if q["id"] not in done]
    if not pending:
        print("All selected questions already completed.")
        if args.evaluate:
            evaluate_output(args)
        return
    from zaic.agent import answer_question
    from zaic.model import HFBackend

    load_start = time.perf_counter()
    backend = HFBackend(
        args.model,
        args.quantization,
        args.device_map,
        args.context_tokens,
        args.max_new_tokens,
        args.thinking,
        args.gpu_memory_gib,
        args.prefill_chunk_size,
        args.temperature,
        args.top_p,
        args.top_k,
        args.repetition_penalty,
        args.prefix_cache,
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
        "quantization_loaded": getattr(backend, "quantization", None),
        # Present when --model is a directory saved by scripts/prepare_model.py.
        "model_source": (
            json.loads((Path(args.model) / "zaic_model.json").read_text())
            if (Path(args.model) / "zaic_model.json").is_file()
            else None
        ),
        "note": "Inference only; no training loss or weight updates.",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    times, errors = [], 0
    stopped_on_error = False
    stopped_on_time = False
    with (
        args.output.open("a", encoding="utf-8") as out,
        trace_path.open("a", encoding="utf-8") as traces,
        metrics_path.open("a", encoding="utf-8") as metrics,
    ):
        for i, q in enumerate(pending, 1):
            hours = (time.perf_counter() - load_start) / 3600
            if args.max_hours and hours >= args.max_hours:
                stopped_on_time = True
                print(
                    f"Time budget of {args.max_hours} h reached after {i - 1} questions; "
                    "rerun with --resume to continue.",
                    flush=True,
                )
                break
            backend.reset_peak()
            if hasattr(backend, "set_seed"):
                backend.set_seed(question_seed(args.seed, q["id"]))
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
                f"{detail['steps']} steps {detail['elapsed_seconds']:.1f}s",
                flush=True,
            )
            print(json.dumps(prediction, ensure_ascii=False), flush=True)
            if detail["status"] == "error":
                stopped_on_error = True
                print(
                    "Stopping after model/runtime failure:", detail["error"], flush=True
                )
                print(f"Full traceback: {trace_path}", flush=True)
                break
    summary = {
        "completed_this_run": len(times),
        "execution_failures": errors,
        "mean_seconds": statistics.mean(times) if times else None,
        "median_seconds": statistics.median(times) if times else None,
        "stopped_on_error": stopped_on_error,
        "stopped_on_time_budget": stopped_on_time,
        "selected_but_not_completed": len(pending) - len(times),
        # Stalled/max_steps questions are normal outcomes and count toward the
        # estimate; only a runtime error (which stops the run) disables it.
        "estimated_3000_questions_hours": (
            statistics.mean(times) * 3000 / 3600 if times and not stopped_on_error else None
        ),
        "estimate_note": "No extrapolation after a runtime error. Otherwise a rough sample estimate excluding model load time.",
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.evaluate:
        evaluate_output(args)
    if stopped_on_error:
        raise SystemExit(1)


def evaluate_output(args):
    from evaluate import evaluate_run

    evaluate_run(
        args.output, args.data_dir / "answers.jsonl", args.data_dir / "questions.jsonl"
    )


if __name__ == "__main__":
    main()
