"""Local deterministic scorer. Does not invoke an LLM or load model weights."""

import argparse
import json
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
        details.append({"id": ident, "outcome": outcome, "expected_type": gold["type"]})

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
    preds, golds = read_jsonl(args.predictions), read_jsonl(args.answers)
    if args.scope == "run":
        manifest = args.predictions.with_suffix(".run.json")
        if not manifest.exists():
            raise SystemExit(
                "Missing .run.json manifest. Use --scope predictions or --scope all explicitly."
            )
        ids = json.loads(manifest.read_text())["config"]["selected_ids"]
    else:
        ids = [v["id"] for v in preds] if args.scope == "predictions" else None
    report = score(
        preds, golds, ids, read_jsonl(args.questions) if args.questions else None
    )
    output = args.output or args.predictions.with_suffix(".evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["overall"], indent=2))
    print(f"Details: {output}")


if __name__ == "__main__":
    main()
