# DataMate solution image: rulebook semantic layer, CPU only, no model weights.
# The organizers mount the test set at /data (read-only) and collect
# /result/submission.jsonl. Nothing is downloaded at run time.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY zaic ./zaic
COPY submit.py inference.py evaluate.py README.md SOLUTION.md ./
COPY scripts ./scripts
COPY tests ./tests

CMD ["python", "submit.py", "--data-dir", "/data", "--output", "/result/submission.jsonl"]
