"""Produce the competition submission file with the semantic layer only.

    python submit.py --data-dir /data --output /result/submission.jsonl

Reads `questions.jsonl` (and `rulebook.md`, `calendar.csv`, `data/<business_id>/`)
from the data directory, answers every question the rulebook code can answer
and writes `{"type": "abstain"}` for the rest. No model is loaded, nothing is
downloaded and the data directory is only read. The output is rewritten after
each business, so a run that is stopped early still leaves a valid file, and
the same input always gives the same output.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from zaic.router import semantic_answers


def find_split(root):
    """The directory that holds questions.jsonl: the mount itself or a folder below it."""
    root = Path(root)
    if (root / "questions.jsonl").is_file():
        return root
    found = sorted(root.glob("*/questions.jsonl")) + sorted(root.glob("*/*/questions.jsonl"))
    if not found:
        raise SystemExit(f"questions.jsonl not found under {root}")
    return found[0].parent


def write(path, questions, answers):
    lines = [json.dumps(answers.get(q["id"], {"id": q["id"], "type": "abstain"}), ensure_ascii=False) for q in questions]
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--output", default="/result/submission.jsonl")
    args = ap.parse_args()
    split = find_split(args.data_dir)
    questions = [json.loads(line) for line in (split / "questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_suffix(".log")
    start = time.time()

    def say(text):
        print(text, flush=True)
        with log.open("a", encoding="utf-8") as f:
            f.write(text + "\n")

    say(f"{len(questions)} questions from {split}")
    by_biz = defaultdict(list)
    for q in questions:
        by_biz[q["business_id"]].append(q)
    answers = {}
    write(out, questions, answers)
    for biz in sorted(by_biz):
        try:
            got = semantic_answers(split, by_biz[biz], log=say)
        except Exception as exc:  # noqa: BLE001 -- one broken shop must not stop the run
            say(f"{biz}: failed ({exc!r}); its questions stay abstain")
            continue
        answers.update({qid: prediction for qid, (prediction, _) in got.items()})
        write(out, questions, answers)
    say(f"answered {len(answers)}/{len(questions)} in {time.time() - start:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
