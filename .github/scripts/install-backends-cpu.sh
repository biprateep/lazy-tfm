#!/usr/bin/env bash
# Installs lazy with every backend package into .venv, with CPU-only PyTorch.
# The export drops the CUDA wheels that uv.lock pins for Linux, and the install
# takes PyTorch from its CPU index. With --upgrade, it first re-locks to the
# newest resolvable versions; the git packages stay at their pinned commits.
set -euo pipefail
if [[ "${1:-}" == "--upgrade" ]]; then
  uv lock --upgrade
fi
uv export --locked --all-extras --group backends --group dev \
  --no-hashes --no-emit-project > requirements.txt
grep -vE '^(nvidia-|triton)' requirements.txt > requirements-cpu.txt
uv venv --python 3.12
uv pip install --torch-backend cpu -r requirements-cpu.txt -e .
