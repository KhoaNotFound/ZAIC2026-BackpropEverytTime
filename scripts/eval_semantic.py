"""Score the deterministic semantic path (rule parser + metrics) on a labeled split.

No model is loaded. For every question the rule parser either produces an
intent (scored against gold) or declines (counted as fallback). Use it to find
data-layer and metric bugs before they reach a GPU run.

    python scripts/eval_semantic.py --data-dir data/train [--business biz_101 ...] [--show wrong]
"""

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import correct
from zaic.intent import entity_index, parse, run
from zaic.metrics import Metrics, Unsupported
from zaic.semantic import Shop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train")
    ap.add_argument("--business", nargs="*")
    ap.add_argument("--show", choices=["wrong", "fallback", "all", "none"], default="wrong")
    ap.add_argument("--limit-show", type=int, default=40)
    ap.add_argument("--output")
    args = ap.parse_args()
    split = Path(args.data_dir)
    questions = [json.loads(line) for line in (split / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    gold = {}
    if (split / "answers.jsonl").exists():
        rows = (split / "answers.jsonl").read_text(encoding="utf-8").splitlines()
        gold = {json.loads(line)["id"]: json.loads(line) for line in rows}
    by_biz = defaultdict(list)
    for q in questions:
        if not args.business or q["business_id"] in args.business:
            by_biz[q["business_id"]].append(q)
    totals = Counter()
    per_metric = defaultdict(Counter)
    per_biz = {}
    shown = Counter()
    records = []
    for biz in sorted(by_biz):
        t0 = time.time()
        try:
            shop = Shop(split, biz)
            metrics = Metrics(shop)
            index = entity_index(shop)
        except Exception:  # noqa: BLE001
            print(f"{biz}: build failed\n{traceback.format_exc(limit=3)}")
            totals["build_failed"] += len(by_biz[biz])
            continue
        c = Counter()
        for q in by_biz[biz]:
            g = gold.get(q["id"])
            try:
                intent = parse(q, shop, split / "calendar.csv", index=index)
            except Exception as exc:  # noqa: BLE001
                intent, err = None, f"parse error {exc!r}"
            else:
                err = None
            if intent is None:
                outcome, pred = "fallback", None
            else:
                try:
                    pred = run(intent, metrics)
                    outcome = "correct" if g and correct(pred, g) else "wrong" if g else "answered"
                except Unsupported as exc:
                    outcome, pred, err = "unsupported", None, str(exc)
                except Exception as exc:  # noqa: BLE001
                    outcome, pred, err = "error", None, f"{exc!r}"
            c[outcome] += 1
            name = intent.metric if intent else "-"
            per_metric[name][outcome] += 1
            records.append({"id": q["id"], "biz": biz, "outcome": outcome, "metric": name, "pred": pred})
            wanted = args.show in (outcome, "all") or (args.show == "wrong" and outcome in ("wrong", "error"))
            if wanted and shown[outcome] < args.limit_show:
                shown[outcome] += 1
                print(
                    f"[{outcome}] {biz} {q['as_of_date']} | {q['question']}\n"
                    f"    intent={intent}\n    pred={pred} gold={g and {k: g.get(k) for k in ('type', 'answer')}} {err or ''}"
                )
        per_biz[biz] = c
        totals.update(c)
        print(f"{biz}: {dict(c)} ({time.time() - t0:.0f}s) issues={shop.issues[:3]}")
    print("\nBy metric:")
    for name, c in sorted(per_metric.items(), key=lambda x: -sum(x[1].values())):
        print(f"  {name:16s} {dict(c)}")
    answered = totals["correct"] + totals["wrong"]
    print(f"\nTotal: {dict(totals)}")
    if answered:
        print(f"Answered {answered}: precision {totals['correct'] / answered:.1%}; score {totals['correct'] - totals['wrong']}")
    if args.output:
        Path(args.output).write_text("\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records), encoding="utf-8")


if __name__ == "__main__":
    main()
