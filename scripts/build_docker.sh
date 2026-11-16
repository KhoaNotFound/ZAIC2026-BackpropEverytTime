#!/usr/bin/env bash
# Build the solution image, test it offline on a local split, and export it
# with a checksum for the "Solution Submission" form.
#   scripts/build_docker.sh [data/public_test]
set -euo pipefail
IMAGE=zaic-datamate:h7
SPLIT=${1:-data/public_test}
OUT=outputs/docker_check
docker build -t "$IMAGE" .
rm -rf "$OUT" && mkdir -p "$OUT"
# Same conditions as the organizers' run: no network, data read-only.
docker run --rm --network none -v "$(pwd)/$SPLIT:/data:ro" -v "$(pwd)/$OUT:/result" "$IMAGE"
wc -l "$OUT/submission.jsonl"
docker save "$IMAGE" | gzip > zaic-datamate-h7.tar.gz
sha256sum zaic-datamate-h7.tar.gz | tee zaic-datamate-h7.tar.gz.sha256
