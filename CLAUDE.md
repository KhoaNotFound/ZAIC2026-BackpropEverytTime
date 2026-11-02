# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DataMate inference pipeline: Qwen3.5 (default `Qwen/Qwen3.5-9B`, NF4 4-bit, split across two Kaggle T4s) answers Vietnamese business questions by calling read-only data tools over per-business Excel/CSV/SQLite files, then emits typed JSON answers. Inference and evaluation only — no training, no weight updates. README.md is in Vietnamese and holds the full Kaggle runbook; AGENTS.md holds repo conventions.

## Commands

Run from the repo root; there is no build step. The local dev machine (WSL) has no `python` on PATH and no project venv; `uv sync` would pull torch/CUDA wheels. For CPU-only testing use an isolated uv env:

```bash
UVX="uv run --no-project --isolated --with pytest --with pandas --with openpyxl --with duckdb --with ruff"
$UVX python -m pytest -q                              # 25 pass; GPU smoke test skips without 2 CUDA GPUs
$UVX python -m pytest -q tests/test_pipeline.py::test_agent_end_to_end_without_gold   # single test
$UVX ruff check zaic inference.py evaluate.py tests

# The commands below are written for Kaggle; locally, prefix with uv as above (or use python3)
# Validate inputs/sampling without loading weights
python inference.py --data-dir data/train --limit 5 --dry-run
# Real run (needs CUDA; --device-map split needs >=2 GPUs; use cuda:0/auto for one)
python inference.py --data-dir data/train --limit 1 --output outputs/smoke/predictions.jsonl
# Custom question (needs --business-id and --as-of-date)
python inference.py --data-dir data/train --business-id biz_123 --as-of-date 2026-06-28 --question "..." --output outputs/custom/predictions.jsonl
# Score (default --scope run reads selected IDs from predictions.run.json; use --scope predictions|all otherwise)
python evaluate.py --predictions outputs/predictions.jsonl --answers data/train/answers.jsonl --questions data/train/questions.jsonl
```

Inference CLI defaults: `--limit 20` (0 = all), `--sampling stratified` (round robin over shuffled businesses; also `random`, `first`), `--seed 42`, `--max-steps 16`, `--context-tokens 16384`, `--max-new-tokens 2048`, `--quantization 4bit` (`8bit`, `none`), `--thinking` off, `--output outputs/predictions.jsonl`.

`data/`, `outputs/`, `reports/` are gitignored. Local splits: `data/train` (has `answers.jsonl`) and `data/public_test` (no answers). Split layout: `{questions.jsonl, answers.jsonl?, rulebook.md, calendar.csv, data/<business_id>/...}`; each business dir must have its own `README.md` (inference validates it up front). Questions are `{id, business_id, question, as_of_date}`, mostly in informal Vietnamese. `scripts/analyze_data.py` profiles the dataset into `reports/` using only the standard library.

## Architecture

Flow: `inference.py` (CLI, sampling, checkpointing) → `zaic.agent.answer_question` (per-question loop) → `QwenBackend.generate` (`zaic/model.py`) and `Tools.call` (`zaic/tools.py`), with parsing/validation in `zaic/formats.py`. `evaluate.py` scores independently.

- **Custom action-JSON protocol, not Qwen native tool calls.** The `SYSTEM` prompt in `agent.py` plus `TOOL_HELP` in `tools.py` define it. Each model turn must be one JSON object: `{"action":"tool","name":...,"arguments":{...}}`, a flat form `{"action":"<tool_name>", ...params}`, or `{"action":"final","type":...,"answer":...}`. `formats.normalize_action` canonicalizes both tool forms (whitelist `TOOL_NAMES`; in the flat form `name` is the load_table *table name*, not the tool). The normalized action replaces the raw assistant message in history so malformed calls aren't reinforced; the raw text stays in the trace. Changing the protocol means updating the SYSTEM prompt, `TOOL_HELP`, `normalize_action`, and the parametrized tests in `test_pipeline.py` together.
- **Agent loop guards** (`agent.py`): non-abstain finals are rejected until at least one tool call succeeded; finals must pass `formats.valid_answer` (types: number, percent, string, date, boolean, list, table, unanswerable+reason, abstain). Three identical consecutive tool calls or three identical (response, error) failures → status `stalled` (prediction stays abstain). Exhausting `max_steps` → `max_steps`. Exceptions → `error` with traceback. Oversized tool output is replaced with a JSON truncation envelope.
- **Tools / data isolation** (`tools.py`): `Tools.resolve` restricts paths to `data/<business_id>/` plus split-level `rulebook.md`/`calendar.csv`, and only `.md/.xlsx/.csv/.sqlite`. `answers.jsonl` must never be reachable by any tool or prompt (tests assert "SECRET LABEL" never appears in prompts). `load_table` loads every cell as text into an in-memory DuckDB (no fill-forward/dedup); tables persist across steps within one question only (a fresh `Tools` per question). `run_sql` is single-statement SELECT on DuckDB with `enable_external_access=False`, 2 threads, 2GB memory and a 30s interrupt timer; `query_sqlite` is read-only with an authorizer and timeout. Results capped at 100 rows. No arbitrary Python execution by design.
- **Model backend** (`model.py`): imports torch/transformers lazily so CLI, evaluator, and tests run without GPU deps. Loads text-only `Qwen3_5ForCausalLM` (requires transformers >=5.13). `make_split_map` builds an explicit device map placing whole decoder blocks contiguously across GPUs (embeddings and `lm_head` on GPU 0, final norm on last GPU). Generation is greedy, fresh dynamic cache per call, `logits_to_keep=1`, chunked prefill (`--prefill-chunk-size`, 0 disables). Raises before generating if input + `max_new_tokens` exceeds `--context-tokens`.
- **Run artifacts & resume** (`inference.py`): from `--output X.jsonl` it writes `X.run.json` (config, selected IDs, source hashes, versions), `X.traces.jsonl`, `X.metrics.jsonl`, `X.summary.json`. The predictions line is written *last* per question and acts as the commit marker. `--resume` requires an identical config/sample/source hashes; otherwise use a new output path (an existing output without `--resume` is refused). A runtime `error` stops the run with exit code 1 and disables time extrapolation.
- **Scoring** (`evaluate.py`): type must match; strings compare after NFC + casefold with whitespace/punctuation stripped (accents are *kept*), against gold plus `alias` (aliases apply to strings, list items by index, and the first table column); numbers need exact match within 1e-8 absolute (deliberately no relative tolerance, so large VND sums must be exact); lists are ordered; tables match columns by name (else position) and rows via bipartite matching; `unanswerable` scores on type only. Score = correct − incorrect; abstain is neutral. If `predictions.metrics.jsonl`/`.run.json` sit next to the predictions, the report also gets a `runtime` benchmark section (status counts, outcome by status, seconds/steps/tool calls/tokens stats, peak VRAM per GPU, load time).

All current work targets the `train` split (the only one with labels). The goal is a baseline of the pretrained, non-fine-tuned Qwen under the current harness; `notebooks/kaggle_inference.ipynb` drives it with `LIMIT`/`SEED` and derives the output dir from them.

## Testing approach

Tests use a synthetic dataset fixture (`source` in `tests/test_pipeline.py`, which plants `answers.jsonl` containing "SECRET LABEL") and a `ScriptedBackend` that replays canned model responses — extend these rather than loading a real model. `tests/test_gpu_smoke.py` builds a tiny random `Qwen3_5TextConfig` model (no download) and checks that split-map + chunked-prefill generation equals single-GPU output. They do not substitute for real Qwen runs on Kaggle; when reporting model changes, distinguish simulated tests from actual checkpoint runs and state hardware/config.

## Conventions

- Commit subjects are imperative ("Accept Qwen flat tool actions and stop repeated protocol loops").
- Keep protocol handling (`formats.py`), data access (`tools.py`), and model execution (`model.py`) in their own modules.
- Never commit datasets, weights, or traces (traces contain business/customer data).
