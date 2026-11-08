# Repository Guidelines

## Problem & Scope

Zalo has a large collection of enterprise data managed by its data team. Employees need to query this data for their work, using natural-language questions like those in `questions.jsonl`. The broader objective is an agent-based system that automates data discovery, querying, and answering with verifiable evidence and appropriate access controls.

Qwen3.5 and the DataMate pipeline are a provisional implementation for experimentation, not the definition of the problem. Do not assume a particular model, dataset format, or Kaggle deployment is a permanent requirement. Evaluate whether each task needs an adaptive agent loop or can use a simpler LLM-assisted query workflow. Prioritize business semantics, source discovery, reliable calculations, permissions, and traceable answers at scale.

## Project Structure & Module Organization

The current Python 3.10+ prototype performs inference and deterministic evaluation without training models.

- `zaic/`: agent loop (`agent.py`), model loading/generation (`model.py`), data tools (`tools.py`), and JSON protocols (`formats.py`).
- `inference.py`: sampling, inference, and run artifacts; `main.py` delegates to it. `evaluate.py` scores predictions independently.
- `tests/`: CPU pipeline tests and an optional GPU smoke test.
- `notebooks/kaggle_run.ipynb`: single Kaggle workflow (prepares the NF4 model and wheels on first run, then runs and scores); `scripts/prepare_model.py`, `scripts/download_wheels.py`, `scripts/check_fast_kernels.py`: its helpers; `scripts/analyze_data.py`: dataset analysis.
- `data/`, `reports/`, and `outputs/`: local datasets and generated artifacts, excluded from Git.

## Build, Test, and Development Commands

Run commands from the repository root. There is no separate build step.

- `pip install -r requirements.txt`: install inference dependencies; preserve compatible CUDA-enabled PyTorch on Kaggle.
- `pip install pandas openpyxl duckdb pytest ruff`: install CPU testing and linting dependencies.
- `python -m pytest -q`: run the test suite.
- `ruff check zaic inference.py evaluate.py tests`: run lint checks.
- `python inference.py --data-dir data/train --limit 5 --dry-run`: validate inputs and sampling without loading weights.
- `python evaluate.py --predictions outputs/predictions.jsonl --answers data/train/answers.jsonl --questions data/train/questions.jsonl`: score a run.

Current inference defaults to two or more CUDA GPUs; see `README.md`.

## Coding Style & Naming Conventions

Use four-space indentation, snake_case functions/modules, PascalCase classes, and UPPER_CASE constants. Keep protocol, data access, and model concerns separate. Lint with Ruff.

## Testing Guidelines

Use pytest, `test_*.py` files, and `test_*` functions with synthetic datasets and scripted responses. Cover protocols, scoring, isolation, and failures. No coverage threshold is configured. `tests/test_gpu_smoke.py` requires two CUDA GPUs; it downloads no weights.

## Commit & Pull Request Guidelines

Use imperative commit subjects, following existing history. PRs should describe behavior changes, relevant issues, and validation results. Distinguish simulated tests from actual model runs.

## Data Boundaries

Preserve business-directory isolation and read-only SQL. Keep answer labels inaccessible to agent tools. Never commit datasets, weights, or traces containing business/customer data. Use a new output path when changing inference configuration.


Competition rounds

The competition runs on a leaderboard with two rounds:

 • Round 1 — Public test: teams develop their solutions, run them on the public test set and upload submission.jsonl. Scores appear on the public leaderboard and are for reference only during development.

 • Final solution submission: before the solution deadline, the final solution is submitted as a Docker image, following the organizers’ separate Docker guideline. For scoring, the organizers mount the test set into the container at /data and collect /result/submission.jsonl.

 • Round 2 — Private test: after teams submit and freeze their solutions, the organizers release the private test set. Teams run their frozen solution on it and upload submission.jsonl. The final ranking is based on the organizers’ re-run of the submitted Docker images (Rule 8).

Technical requirements

Every solution must meet all of the following requirements:

1. Teams and eligibility follow the general regulations of Zalo AI Challenge 2026.
2. Models. You may use any model whose weights were publicly released before 1 October 2026 under a licence that permits such use, and you may fine-tune it freely. Across all models used at inference by your solution (language models, embedding models, rerankers, classifiers), whether loaded at the same time or not, the total number of parameters must not exceed 15 billion, counting all parameters of mixture-of-experts models. Any quantization is allowed. The models used must be declared in the solution README.
3. No external services. Inference must run fully offline inside your Docker image: no calls to commercial LLM APIs, no downloads, no network access of any kind. The image is run with networking disabled.
4. No human in the loop. Answers must be produced by your system without manual intervention. Hard-coding answers, or logic keyed on question or business identifiers, is prohibited.
5. Code execution. Your agent may generate and execute code inside the container; this is expected but not required. The data directory is mounted read-only; write temporary files elsewhere.
6. Data. You may use the provided training data, data you synthesize yourself, and publicly available datasets and models. You may not use the public or private test data for training, nor collect answers to test questions by any means.
7. Compute reference. Solutions are verified on a machine with 2 × NVIDIA RTX 5090 (32 GB VRAM each, 64 GB total), 512 GB RAM. Your solution must process the entire private test set within 6 hours of wall-clock time on this machine, including model loading. Blackwell GPUs require CUDA ≥ 12.8 and a matching PyTorch/vLLM build; the Docker guideline specifies the base image to use.
8. Final score and reproducibility. The score of the organizers’ re-run of your submitted Docker image on the private test set is your final score. The submission.jsonl you upload during the private test phase determines only the provisional leaderboard and which teams are re-run. Your solution must be deterministic enough that the two runs agree: a solution is considered reproducible if |PA(uploaded) − PA(re-run)| ≤ 0.03. If the difference exceeds 0.03, the organizers will examine the logs written by your image (see Submission format, Section Data), may run the image once more, and may request additional information from the team; if the difference cannot be explained, the entry is disqualified. If the re-run exceeds the time limit of Rule 7 or terminates early, only the answers written to /result/submission.jsonl before termination are scored; the rest count as abstain.
