# T4 language experiment: frequency collapse, not useful context learning

## Scientific result

The model trained on real WikiText-2 text, but did **not** demonstrate useful
context-conditioned prediction in this run. All three trained real-connectome
seeds and all three trained rewired seeds reached the same held-out accuracy
as the most frequent character, space. Autoregressive output was almost entirely
spaces.

The fixed experiment used 1000 updates, 128 sampled windows per update, 16
context characters, 48 output characters, and seeds 0, 1, 2. Each neural score
uses 2048 non-overlapping held-out windows; the baselines use exactly the same
target positions. No hyperparameters were changed after examining these results.

| Model | Initial test accuracy | Final test accuracy | Final test BPC |
|---|---:|---:|---:|
| Real topology, reward learning | 2.02% | 18.51% | 13.41 |
| Rewired topology, reward learning | 4.52% | 18.51% | 13.55 |
| Real topology, frozen weights | 2.02% | 2.02% | 5.81 |
| Train-count unigram | -- | 18.51% | 4.52 |
| Train-count bigram | -- | 29.35% | 3.46 |
| Train-count trigram | -- | 39.70% | 2.92 |

Neural entries are means across three seeds. Uniform random sampled accuracy
would be 2.08%. Greedy unigram prediction always chooses space; its BPC uses the
smoothed unigram probability distribution, not a zero-entropy constant policy.
Neural BPC is conditional on sampled hidden trajectories, not exact marginal
hidden-state likelihood.

Both trained controls have zero between-seed SD in final greedy accuracy because
each scores exactly 379/2048. Their mean expected sampled reward is approximately
18.46%, consistent with a nearly deterministic frequent-character policy.

For the three real-graph generated continuations, **359 of 360 generated
characters were spaces**:

| Seed | Prompt | Generated suffix |
|---|---|---|
| 0 | `= homarus gammar` | 120 spaces |
| 1 | `= homarus gammar` | 25 spaces, `(`, 94 spaces |
| 2 | `= homarus gammar` | 120 spaces |

The literal outputs are preserved in the run JSON files. These short samples
do not exhaust all possible prompts, but they corroborate the frequency-collapse
interpretation rather than meaningful text generation.

BPC worsened from 5.81 to 13.41 for the real graph. Thus reward increased while
probability assignment to less frequent correct characters became worse.
Correctness-only reward does not enforce calibrated language probabilities.
It also does not make constant output globally optimal: the n-gram baselines
demonstrate that these targets support substantially better conditional choices.
This run cannot isolate whether credit-assignment variance, exploration,
neural dynamics, capacity, or their interaction caused the collapse.

## What was verified

- Actual **Tesla T4** execution; every reported neural run used `cuda:0`.
- Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8.
- 209 neurons and 7425 directed anatomical connections, represented by dense
  masked float32 GPU matrices.
- Fixed edge-independent input/output assignments shared across controls.
- Source-normalized initial weight values, preserved anatomical mask and signs.
- Reward from the action actually sampled; no label is passed into the rollout,
  no autograd/BPTT, and no learned external encoder or decoder.
- Separate official text splits, train-only vocabulary and n-gram fitting.
- **63 tests passed on Colab in 11.28 seconds.**
- All three seed-zero controls matched uninterrupted execution after resuming
  from update 500 to update 1000: checkpoint arrays and metadata, final metrics,
  reward history, final weight hashes, and generated continuations were identical.
- Fresh full Ruff and strict basedpyright checks passed; wheel/sdist builds and
  `bash -n run_language_colab.sh` passed.

The resumed main experiment covers nine runs. The independent uninterrupted
resume reference covers three runs, all for seed zero; it does not claim an
additional uninterrupted reference for seeds one and two.

GPU experiment and resume verification wall time was **228.36 seconds**,
excluding setup, transfers, and the test run. Maximum allocated PyTorch CUDA
tensor memory was **106,068,480 bytes**, about 101.2 MiB; this is not total GPU
process/context memory or a GPU-utilization measurement. The T4 was released
after results were downloaded.

The persistent LSP daemon retained stale missing-import diagnostics. A fresh
complete checker resolved the project with zero errors or warnings. Bash LSP
was unavailable; the shell runner was checked with `bash -n` and its constituent
Colab operations were exercised.

## Reproduce and inspect

Run `bash run_language_colab.sh`; see `LANGUAGE.md` for configuration and caveats.

- `results/language-t4/resumed/`: all nine final per-run JSON files and NPZ
  checkpoints, plus the combined summary.
- `results/language-t4/uninterrupted-seed0/`: independent resume references.
- `results/language-t4/evidence.json`: actual device, timing, and match count.
- `results/language-t4/tests.txt`: remote test output.
- `data/wikitext2/baselines.json`: matched unigram/bigram/trigram references.
- `data/wikitext2/provenance.json`: exact source and preprocessing identity.
- `artifacts/flyrl-language-results.zip`: original downloaded remote results.
- `artifacts/flyrl-language-source.tar.gz`: exact uploaded execution source.

Source archive SHA-256:

```text
a0a915c1ab3cec8873178c751d3a5bbc1a61690afc85c99aad5082c47b7c9094
```

The result does not support a fly-topology advantage or a claim of learned
grammar. The next decision should address this frequency-collapse failure mode
with a controlled learning/dynamics experiment, rather than simply increasing
GPU size or reporting improvement over uniform random as language acquisition.
