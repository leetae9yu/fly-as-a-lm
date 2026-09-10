#!/usr/bin/env bash
# Run from a machine with authenticated colab CLI access. CPU only.
set -euo pipefail
cd "$(dirname "$0")"
session="${FLYRL_SESSION:-flyrl-$(date +%s)}"
mkdir -p artifacts
tar --exclude='__pycache__' -czf artifacts/flyrl-source.tar.gz \
    flyrl scripts tests data pyproject.toml uv.lock
colab new -s "$session"
trap 'colab stop -s "$session"' EXIT
colab upload -s "$session" artifacts/flyrl-source.tar.gz /content/flyrl-source.tar.gz
colab install -s "$session" \
    numpy==2.5.3 pydantic==2.13.5 typer==0.27.2 pytest==9.1.1
colab exec -s "$session" -f scripts/colab_run.py --timeout 900
colab download -s "$session" /content/flyrl-results.zip artifacts/flyrl-results.zip
