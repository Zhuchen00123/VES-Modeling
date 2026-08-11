#!/usr/bin/env bash
# Build the candidate sandbox image with numpy/pandas/scikit-learn preinstalled.
set -euo pipefail
IMAGE="${1:-ves-modeling-runner:0.1}"
docker build -t "$IMAGE" - <<'DOCKERFILE'
FROM python:3.13-slim
RUN pip install --no-cache-dir numpy pandas scikit-learn
DOCKERFILE
echo "built $IMAGE"
