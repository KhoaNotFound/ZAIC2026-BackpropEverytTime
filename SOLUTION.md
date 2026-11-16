# DataMate solution (harness h7)

## What the image does

`python submit.py --data-dir /data --output /result/submission.jsonl`

For every business in the test set the code reads the README and the data
files, builds clean tables (orders, order lines, purchases, payments, stock)
and answers each question with rulebook arithmetic written in Python
(`zaic/`). A question is answered only when the rule-based parser accounts for
every word of it; otherwise the line is `{"type": "abstain"}`.

- Models used at inference: **none** (0 parameters). No LLM, embedding model
  or reranker is loaded.
- No network access, no downloads, no external services. The data directory
  is only read; the only files written are `/result/submission.jsonl` and
  `/result/submission.log`.
- Deterministic: the same input gives the same output, so the organizers'
  re-run reproduces the uploaded file.
- No logic is keyed on question or business identifiers; column roles come
  from each README and from generic header vocabulary.
- CPU only. About 3.5 minutes for the 1000-question public test on a laptop;
  the output file is rewritten after each business, so an interrupted run
  still leaves a valid file.

## Input layout

`/data` holds `questions.jsonl`, `rulebook.md`, `calendar.csv` and
`data/<business_id>/...` (the layout of the released splits). If
`questions.jsonl` is one or two folders below `/data`, it is found
automatically.

## Build, test, export

```bash
scripts/build_docker.sh data/public_test   # builds, runs offline, writes the .tar.gz and its sha256
```

Manual run:

```bash
docker run --rm --network none -v /path/to/test:/data:ro -v /path/to/result:/result zaic-datamate:h7
```

## Code map

- `submit.py` — entry point of the image.
- `zaic/readme.py`, `zaic/semantic.py`, `zaic/values.py` — README parsing, file reading, cell parsing, clean tables.
- `zaic/periods.py`, `zaic/intent.py`, `zaic/lexicon.py` — time phrases, question parsing, known vocabulary.
- `zaic/metrics.py` — rulebook metrics.
- `inference.py`, `zaic/agent.py`, `zaic/model.py`, `zaic/tools.py` — the optional Qwen3.5-9B agent used during development; it is not run by the image.
- `scripts/eval_semantic.py` — scores the semantic layer on the train split. There is no training step: nothing is learned from data.
