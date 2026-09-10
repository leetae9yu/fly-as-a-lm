#!/usr/bin/env bash
# Authenticated colab CLI required. Requests only a free-tier-eligible T4.
set -euo pipefail
cd "$(dirname "$0")"
session="${FLYRL_SESSION:-flyrl-language-$(date +%s)}"
mkdir -p artifacts
tar --exclude='__pycache__' -czf artifacts/flyrl-language-source.tar.gz \
    flyrl scripts tests data pyproject.toml uv.lock
colab new -s "$session" --gpu T4
trap 'colab stop -s "$session"' EXIT
colab upload -s "$session" \
    artifacts/flyrl-language-source.tar.gz /content/flyrl-language-source.tar.gz
# Keep Colab's CUDA-enabled PyTorch; never install a CPU wheel on the GPU runtime.
colab install -s "$session" \
    numpy==2.5.3 pydantic==2.13.5 typer==0.27.2 pytest==9.1.1
colab restart-kernel -s "$session"
colab exec -s "$session" -f scripts/t4_bootstrap.py --timeout 1800
colab download -s "$session" \
    /content/flyrl-language-results.zip artifacts/flyrl-language-results.zip
