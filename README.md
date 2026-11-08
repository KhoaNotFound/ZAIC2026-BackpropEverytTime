# DataMate — Qwen3.5 on Kaggle

An inference pipeline for business data in multiple formats. Qwen3.5 reads the README and rulebook, selects tools, calculates results with SQL, and returns JSON. Excel, CSV, and SQLite sources remain unchanged; intermediate tables are created in memory for each question. The pipeline does not fine-tune models, update weights, or compute training loss.

## Running on Kaggle (training-split benchmark)

The goal is to measure how many questions the pretrained Qwen3.5-9B model, without fine-tuning, can answer correctly on the **training split** with the current harness. The benchmark workflow uses `train` because it includes `answers.jsonl` for evaluation.

### 1. Prepare the notebook

1. Create a new Kaggle notebook. In **Session options** on the right:
   - **Accelerator**: select **GPU T4 x2** (two GPUs are required for the default configuration).
   - **Internet**: turn it **On** to clone the repository (and, without the prepare notebook's output, to download Qwen weights).
2. Attach the dataset: **Add Input** → **Your Work** / **Datasets** → select your uploaded dataset → **Add**.
3. The dataset must contain a training directory with this structure:

```text
.../train/
  questions.jsonl
  answers.jsonl
  rulebook.md
  calendar.csv
  data/biz_101/README.md ...
```

Do not place `answers.jsonl` inside a business directory. Datasets are excluded from Git (`data/` is ignored).

### 2. Find the dataset path

Kaggle mounts datasets read-only under `/kaggle/input/`. The directory name is the dataset slug. Additional directory levels may depend on how you uploaded the data, such as a ZIP archive containing `data/train`. When `DATA_DIR = ""`, the notebook **automatically finds** a directory containing `questions.jsonl`, `answers.jsonl`, `rulebook.md`, and `data/`. To inspect the paths yourself, run this cell:

```python
!ls /kaggle/input
!find /kaggle/input -maxdepth 4 -name questions.jsonl
```

The training path is the directory containing `questions.jsonl`, for example `/kaggle/input/<slug>/train`. You can also select the directory in the **Input** panel on the right and use the copy icon to copy its path.

### 3. Prepare the model and libraries once (`kaggle_prepare.ipynb`)

Import [notebooks/kaggle_prepare.ipynb](notebooks/kaggle_prepare.ipynb) (**File → Import Notebook**), select a **GPU T4** accelerator with **Internet On** (no dataset needed), and run it with **Save Version → Save & Run All (Commit)**. It takes roughly 10–20 minutes and writes about 7 GB to `/kaggle/working`:

- `qwen3.5-9b-nf4/`: the text-only Qwen3.5-9B decoder already quantized to NF4 with exactly the settings inference uses online (`zaic.model.quantization_config`), plus the tokenizer and a `zaic_model.json` provenance file. It is produced by `scripts/prepare_model.py`, which reloads the saved directory as inference does and generates once (`--verify`).
- `wheels/`: the Python packages the Kaggle image is missing (`scripts/download_wheels.py` asks pip which packages `requirements.txt` would add, skipping torch and CUDA libraries), installed offline with `pip install --no-index --no-deps`.

Rerun it only when changing the model or quantization, or when Kaggle updates its image (the inference notebook falls back to an online `pip install` if the wheels no longer fit).

### 4. Run the benchmark (`kaggle_inference.ipynb`)

Import [notebooks/kaggle_inference.ipynb](notebooks/kaggle_inference.ipynb), select **GPU T4 x2**, and attach two inputs: the dataset with the training split, and **Add Input → Your Work → Notebooks →** the prepare notebook. Internet is only needed to clone the repository. Only the first cell needs editing:

1. **Configuration**: `LIMIT`, `SEED`, `CONTEXT_TOKENS`, `TEMPERATURE`, `HARNESS`, `RESUME`. `DATA_DIR`, `MODEL_DIR` (a directory containing `zaic_model.json`) and the wheel directory are discovered under `/kaggle/input`. Without a prepared model it falls back to downloading `Qwen/Qwen3.5-9B` and quantizing while loading, which is much slower. `OUTPUT` encodes model, harness, `LIMIT`, `SEED`, context and temperature, so changed configurations never collide.
2. **Setup**: clone or pull the repository, install the prepared wheels offline, print library versions and GPUs. Inference runs in a subprocess, so no kernel restart is needed after installing.
3. **Run**: a dry run validates the data without loading the model, then `inference.py --evaluate` runs the sample and scores it immediately. With a local model, `HF_HUB_OFFLINE=1` prevents any Hugging Face access. Logs show the device map, per-GPU VRAM after loading, and `[i/n] id status type ...` per question.
4. **Results**: prints the summary and the benchmark without per-question details (those are in `predictions.evaluation.json`).

The benchmark includes correct, incorrect, and abstained counts; accuracy; coverage; attempted accuracy; score (correct +1, incorrect −1, abstain 0); results by answer type; termination statuses such as `ok`, `stalled`, `max_steps`, and `error`; outcomes by status; time, steps, tool calls, prompt, prefilled and output tokens per question; peak VRAM per GPU; and model loading time. `predictions.run.json` records `model_source` (from `zaic_model.json`) and `quantization_loaded`.

### 5. Suggested workflow

1. `LIMIT = 1`: check that the model loads and inspect peak VRAM per GPU in the benchmark.
2. `LIMIT = 20` (or 50, 100): estimate initial accuracy. Update the first cell, then rerun all cells.
3. `LIMIT = 0`: run the full training split. The time estimate for 3,000 questions is in `predictions.summary.json`. Kaggle GPU session and weekly usage limits may require multiple sessions with `RESUME = True`.
4. If a session is interrupted, keep the same configuration in Cell 1, set `RESUME = True`, and rerun. Questions with saved predictions are skipped.
5. Use **Save Version**, or download the output directory under `/kaggle/working/`, to preserve results before the session ends.

**Context:** the system prompt, `rulebook.md` (approximately 36 KB), and business README already use roughly 12,000–13,000 tokens before any tool calls. Both `inference.py` and the notebook therefore default to 32,768 tokens. If a question exceeds the limit, the harness abstains with status `context_limit` and continues to the next question. If many questions reach `context_limit`, increase the budget to 49,152 or 65,536 and monitor peak VRAM. Longer context also slows each step because the entire prompt is processed again.

If execution stops with an error, such as CUDA out of memory, the program saves the traceback to `predictions.traces.jsonl` and exits with code 1. Restart the kernel to release GPU memory before trying again with a new output path and an adjusted configuration if needed.

### Run from the command line

Check question selection without loading the model:

```bash
python inference.py --data-dir /kaggle/input/<dataset>/train --limit 5 --dry-run
```

Run one question first to check model loading and VRAM, then use a new output path for a 20-question run:

```bash
python inference.py \
  --data-dir /kaggle/input/<dataset>/train \
  --model Qwen/Qwen3.5-9B --quantization 4bit \
  --device-map split --limit 1 --sampling stratified \
  --output /kaggle/working/smoke/predictions.jsonl
```

By default, `--device-map split` distributes whole decoder layers across all visible GPUs. For Qwen3.5-9B on two T4s, the first 16 layers are placed on GPU 0 and the remaining 16 on GPU 1. Embeddings are on GPU 0, the final norm is on the last GPU, and the output head always shares GPU 0 with the input. This mode requires at least two GPUs. It uses model parallelism, rather than two workers processing separate questions or tensor parallelism, and does not guarantee twice the speed.

Other defaults are FP16 compute, NF4 4-bit quantization, a 32,768-token context budget including input and output, up to 2,048 output tokens per turn, and 16 turns per question. The backend (`HFBackend`) loads any Hugging Face causal LM with a chat template through `AutoModelForCausalLM`; Qwen3.5 checkpoints use the text-only `Qwen3_5ForCausalLM` class, omitting the unused vision tower. To benchmark another model, change only `--model` (and its recommended sampling values); prompts, tools, loop rules and scoring stay identical. Chat templates that reject a system role get the system prompt folded into the first user turn. It computes logits only for the final token (`logits_to_keep=1`) and processes prompts in 512-token chunks by default (`--prefill-chunk-size 512`; set it to 0 to disable chunking). Chunking preserves the full prompt and requires Transformers >=5.13 to maintain hybrid attention state across chunks. The backend reuses prompt prefixes (`PrefixCache`): the system message (protocol, tool help and rulebook, identical for every question) is prefilled once per run, and within a question each step only prefills what is new since the previous user message. Hybrid linear-attention caches cannot be cropped, so states are snapshotted at message boundaries and deep-copied before each generation; a boundary is used only if re-tokenizing the text up to it reproduces the prompt's tokens exactly, so the model input never changes. `cached_tokens` in the trace/metrics and `prefilled_tokens_per_question` in the evaluation runtime show the savings. If anything in that path fails, the backend prints a warning and falls back to full prefill; `--no-prefix-cache` disables it. The code prints the device map and per-GPU memory after loading. Evenly distributing layers can still produce uneven VRAM usage because of embeddings, the output head, caches, and temporary tensors.

`--device-map auto`, `balanced`, `balanced_low_0`, and `sequential` are also supported. `--gpu-memory-gib 13` sets a weight-placement budget for these automatic mapping modes, not a hard limit on total VRAM. It does not apply to the explicit `split` map. `--device-map cuda:0` uses one GPU. `--quantization none` uses FP16 and requires more VRAM. Hybrid attention may fall back to a slower PyTorch implementation when suitable accelerated kernels are unavailable; do not install FlashAttention on T4 without checking compatibility.

If using uv on Kaggle, update the repository and run:

```python
%cd /kaggle/working/zAIC
!git pull --ff-only
!uv sync --upgrade-package transformers
!uv run python inference.py --data-dir "<actual training directory>" --device-map split --quantization 4bit --limit 5 --output /kaggle/working/split/predictions.jsonl
```

Stop any previous inference process first. If the model remains loaded in the notebook kernel, restart the session or kernel to release GPU memory. When changing the device map, use a new output path instead of resuming the previous run. Splitting distributes weights, but long context or other processes using VRAM can still cause out-of-memory errors. The split and chunked-prefill changes have not yet been benchmarked on actual T4 hardware. If errors persist, inspect `error_traceback` in the traces or metrics to identify the failing operation.

If context is exceeded, the harness first replaces the oldest full tool results with a short stub (counted as `compactions` in the metrics; the latest result, the rulebook and the README are never dropped) and retries. Only when nothing is left to drop does it abstain with status `context_limit` and continue to the next question. Inspect the trace, increase `--context-tokens` if VRAM permits, or reduce tool output size or the number of processing steps. `--thinking` enables internal reasoning and may require a larger `--max-new-tokens` budget; it is disabled by default for a faster baseline. Generation defaults to Qwen's non-thinking preset (`--temperature 0.7 --top-p 0.8 --top-k 20`). Each question's sampling seed is derived from `--seed` and its ID to support reproducible reruns and `--resume` on the same hardware. Greedy generation (`--temperature 0`) is prone to repetitive token loops. Generation stops as soon as the model completes a JSON object, so `--max-new-tokens` is only a safety ceiling.

### Evaluate results

```bash
python evaluate.py \
  --predictions /kaggle/working/smoke/predictions.jsonl \
  --answers /kaggle/input/<dataset>/train/answers.jsonl \
  --questions /kaggle/input/<dataset>/train/questions.jsonl
```

By default, evaluation covers exactly the IDs in the inference run's `.run.json` manifest. Selected questions that were not processed count as abstentions. A five-question run is therefore scored over those five questions, not all 3,000. Use `--scope predictions` to score only IDs present in an external predictions file, or `--scope all` to score all labeled questions.

The report includes accuracy over the full sample, coverage, accuracy among attempted questions, the **correct +1 / incorrect −1 / abstain 0** score, and breakdowns by answer type and business. This is a local evaluator based on the released rulebook and aliases, not the official scorer. Valid name variants found in source data but missing from the aliases may need review. An `unanswerable` reason must be nonempty, but it is not graded by an LLM.

### Resume or run the full split

Reuse the original inference command and add `--resume`. The configuration and selected question list must match. Only IDs with saved predictions are skipped. Model or runtime errors are saved as abstentions with a traceback, and the program immediately exits with code 1. To retry failed questions, start a new run. The harness does not continue using the model after an out-of-memory error or extrapolate runtime after execution errors. Use a new output path when changing the limit, model, or other configuration.

```bash
python inference.py \
  --data-dir /kaggle/input/<dataset>/train --limit 0 \
  --output /kaggle/working/full/predictions.jsonl
```

`--limit 0` selects all questions. Default sampling uses round robin across shuffled businesses and does not guarantee that small samples represent every business task category. The extrapolated time for 3,000 questions is only a rough estimate from the current sample and excludes model loading. Kaggle sessions may be interrupted. Save a notebook version or download outputs to preserve artifacts.

### Ask a custom question

```bash
python inference.py \
  --data-dir /kaggle/input/<dataset>/train \
  --business-id biz_123 \
  --question "What was the total revenue in May 2026?" \
  --as-of-date 2026-06-28 \
  --output /kaggle/working/custom/predictions.jsonl
```

### Output files

- `predictions.jsonl`: only ID, type, and answer or reason, in submission format.
- `predictions.run.json`: configuration, selected questions, library versions, device map, and model loading time.
- `predictions.traces.jsonl`: model responses, tool calls, results, and errors.
- `predictions.metrics.jsonl`: per-question time, tokens, and VRAM without the full trace.
- `predictions.summary.json`: timing statistics for the current session's execution; extrapolations are null if execution errors occurred.
- `predictions.evaluation.json`: the benchmark produced by `evaluate.py`, including overall accuracy, coverage, score, breakdowns by answer type and business, and `runtime`. Each `details` entry contains the question text when supplied, the scored `prediction`, the gold `expected` answer, `outcome` (correct/incorrect/abstained), `points` (+1/−1/0), and `harness_status` when metrics are available. Missing predictions are recorded as `null` and count as abstentions.

Inference logs print the answer after each question. `ok number` means the agent returned a valid final answer of type number; it does not establish that the answer is correct. To compare against `answers.jsonl` and print scores and per-question results immediately after inference:

```bash
python inference.py --data-dir data/train --limit 20 \
  --output outputs/benchmark/predictions.jsonl --evaluate
```

`--evaluate` requires dataset questions with answer labels. It can also be added when resuming a completed run with `--resume`. Only the evaluator reads gold answers after inference; they are never passed to the model or its tools. The evaluator scores the saved predictions without replacing them with SQL results or correcting model answers. If the harness does not receive a valid final answer, the prediction is an abstention; the original response and rejected final answer remain in the trace.

Outputs are saved after each question to support resuming. If the process stops partway through writing a JSONL record, repair or remove the incomplete final line before resuming. The parser reports malformed data rather than skipping it. Traces may contain business and customer information and need the same access restrictions as the source data.

## Tools and data boundaries

- `list_files`, `read_document`, `inspect_table`: discover data within the current business. `inspect_table` lists every sheet with its real row count and previews the first 6 rows (numbered from 0) of as many sheets as fit; `"sheet"` previews one sheet.
- `load_table`: load a CSV, one Excel sheet, a list of sheets, or all sheets (`"sheet":"*"`, stacked and matched by header name) as one DuckDB table, with a selectable zero-based header row. Cells are text exactly as `inspect_table` shows them (Excel dates as `YYYY-MM-DD[ HH:MM:SS]`, integral numbers without `.0`); blank cells are NULL; fully blank rows are skipped. `_row` (source row index) and, for Excel, `_sheet` keep the source order so the rulebook's group fill-down and duplicate rules can be applied in SQL. Nothing is filled, deduplicated or converted automatically.
- `run_sql`: execute DuckDB SELECT queries over loaded tables, blocking external file and network access.
- `query_sqlite`: query a read-only connection with a read-only authorizer and a timeout.

Paths are restricted to the current business directory plus the split's `rulebook.md` and `calendar.csv`. The agent has no tool for reading JSONL files or answer labels. Only the evaluator reads `answers.jsonl`. Tools return at most 100 rows, but this does not limit the rows included in aggregate calculations. Large results are marked as truncated.

The model uses a **custom JSON action protocol**, rather than Qwen's native tool-call parser. Both wrapped actions such as `{"action":"tool","name":"load_table","arguments":{...}}` and flat actions such as `{"action":"load_table","path":"...","name":"sales"}` are accepted; tool names must be allowlisted. For `list_files`, an extra path argument is ignored because the tool is always scoped to the current business. Traces retain both the raw response and `normalized_action`. Prompts include tool-call examples with actual source filenames, and the harness supports retries to correct protocol errors. JSON is extracted from the first complete top-level object, ignoring surrounding text. Unknown keys such as `thought` are discarded, while missing arguments produce explicit errors.

Error feedback depends on the error type: SQL errors include loaded tables and columns, path errors include available files, and invalid final-answer types include the required format. Long error feedback is shortened in conversation history while remaining complete in the trace. If a response exactly repeats the previous one, the harness warns the model and does not execute it again. A third consecutive repetition produces `stalled` and an abstention. Final answers, except abstain and unanswerable, require at least one successful data-tool call; `list_files` does not count. Following rulebook §7, `reason` is optional for unanswerable. Every feedback message states how many actions remain, and the prompt states the step budget. SQL DECIMAL results are returned as JSON numbers. `HARNESS_VERSION` (in `zaic/agent.py`) is stored in `.run.json`; bump it whenever prompts, tools or loop rules change.

Arbitrary Python execution is not currently available. Normalization uses SQL and table-loading parameters. Some businesses may require additional tools with deterministic processing functions. Timeouts currently apply to SQL queries. Workbook loading and generation are bounded by resources, context, and token limits, but there is no hard timeout for an entire question.

## Testing without a GPU

```bash
pip install pandas openpyxl duckdb pytest ruff
python -m pytest -q
ruff check zaic inference.py evaluate.py tests
```

The latest local validation passed 47 tests (46 CPU pipeline tests plus the prefix-cache equivalence test on a tiny random Qwen3.5 model, run on CPU with PyTorch 2.15 and Transformers 5.22) and Ruff checks. One GPU test was skipped; it requires PyTorch and two CUDA GPUs. The small GPU test downloads no checkpoint and compares single-GPU generation against split placement with chunked prefill: `uv run pytest -q tests/test_gpu_smoke.py`.

Initial data checks were performed in the development environment with Python 3.13, pandas 3.1.0, DuckDB 2.0.0, and OpenPyXL 3.1.5.

Tests use synthetic sources and scripted model responses to cover the agent loop, tools, answers, evaluation, file isolation, read-only SQL, aliases, list and table answers, sampling, and formats. They do not replace actual Qwen testing on Kaggle.

## Model documentation

- [Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Transformers Qwen3.5](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [BitsAndBytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

Python >=3.10 and Transformers >=5.13 are required for `Qwen3_5ForCausalLM` and hybrid-attention chunked prefill. After a successful Kaggle run, save `pip freeze` alongside the notebook to record the validated environment.

Device-map strategy reference: [Accelerate Big Model Inference](https://huggingface.co/docs/accelerate/concept_guides/big_model_inference).

Chunked prefill: [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation). Text-only Qwen loader: [Qwen3.5 docs](https://huggingface.co/docs/transformers/v5.13.0/model_doc/qwen3_5).
