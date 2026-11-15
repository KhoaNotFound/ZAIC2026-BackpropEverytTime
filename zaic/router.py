"""Answer questions through the semantic layer before any model call.

`semantic_answers(split_dir, questions)` builds each business's Shop once
(questions grouped by business, one shop in memory at a time) and returns
{question id: (prediction, detail)} for every question the rule parser and
metrics can answer. Questions it cannot answer are left for the agent.
"""

import time
import traceback
from collections import defaultdict
from pathlib import Path

from .intent import entity_index, parse, run
from .metrics import Metrics, Unsupported
from .semantic import Shop


def semantic_answers(split_dir, questions, log=print):
    split_dir = Path(split_dir)
    calendar = split_dir / "calendar.csv"
    by_biz = defaultdict(list)
    for q in questions:
        by_biz[q["business_id"]].append(q)
    out = {}
    for biz, qs in by_biz.items():
        start = time.perf_counter()
        try:
            shop = Shop(split_dir, biz)
            metrics = Metrics(shop)
            index = entity_index(shop)
        except Exception:  # noqa: BLE001 -- a broken shop only disables the fast path
            log(f"semantic: {biz} build failed, using the agent\n{traceback.format_exc(limit=3)}")
            continue
        built = time.perf_counter() - start
        answered = 0
        for q in qs:
            t0 = time.perf_counter()
            try:
                intent = parse(q, shop, calendar if calendar.is_file() else None, index=index)
                if intent is None:
                    continue
                prediction = run(intent, metrics)
            except Unsupported:
                continue
            except Exception as exc:  # noqa: BLE001 -- unexpected: leave it to the agent
                log(f"semantic: {q['id']} failed ({exc!r}), using the agent")
                continue
            answered += 1
            out[q["id"]] = (
                {"id": q["id"], **prediction},
                {
                    "id": q["id"],
                    "business_id": biz,
                    "status": "semantic",
                    "error": None,
                    "error_traceback": None,
                    "elapsed_seconds": time.perf_counter() - t0,
                    "steps": 0,
                    "generations": 0,
                    "resampled": 0,
                    "tool_calls": 0,
                    "compactions": 0,
                    "input_tokens": 0,
                    "cached_tokens": 0,
                    "output_tokens": 0,
                    "intent": {
                        "metric": intent.metric,
                        "period": [str(intent.period.start), str(intent.period.end)],
                        "base": [str(intent.base.start), str(intent.base.end)] if intent.base else None,
                        "filters": {k: str(v) for k, v in intent.filters.items()},
                        "k": intent.k,
                        "measure": intent.measure,
                    },
                    "shop_build_seconds": built,
                    "trace": [],
                },
            )
        log(f"semantic: {biz} answered {answered}/{len(qs)} (shop built in {built:.0f}s)")
        del shop, metrics, index
    return out
