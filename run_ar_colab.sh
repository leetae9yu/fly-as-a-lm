#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m flyrl.autoregressive \
  --graph data/large_connectome/malecns_v1_n16384.npz \
  --corpus data/ar_corpus/corpus.npz \
  --output results/ar-main \
  --device cuda \
  --updates 4000 \
  --batch-size 8 \
  --context 32 \
  --seeds 0 \
  --controls real,shuffled,frozen \
  --learning-rate 0.003 \
  --eval-windows 1024 \
  --sample-length 240 \
  --checkpoint-steps 200 \
  --readout-neurons 256 \
  --edge-chunk 65536 \
  --progress \
  "$@"
