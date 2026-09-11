# Full-training BPE comparison protocol

**This is the planned protocol, not a record of 15 completed runs.** The study
stopped after six complete runs and one interruption. See
[BPE_RESULTS.md](BPE_RESULTS.md) for the actual coverage and measurements.

This is a resource-limited, fixed-budget comparison, not training to convergence.
The immutable settings are in [BPE_STUDY.json](BPE_STUDY.json), committed as
`548400e` before the first main training job. The budget was selected from T4
throughput measurements, not held-out scores.

## Data and evaluation

The byte-level BPE vocabulary has 4,096 entries and is fitted only on all
normalized WikiText-2 training text. The normalization remains lowercase plus
collapsed whitespace. This is the tokenized upstream dataset: its literal
`<unk>` markers cannot recover words already lost upstream.

| Split | Normalized character bounds | BPE tokens |
| --- | --- | ---: |
| Train | `[0:10707000]` | 2,830,508 |
| Validation | `[0:1112671]` | 288,823 |
| Test | `[196608:1246301]` | 276,241 |

The test prefix through character 196,608 is excluded because earlier
experiments used regions in that prefix. The new corpus and tokenizer do not
overwrite the character pilot or the short BPE implementation check.

Each score evaluates the final target of **1,024 identical, non-overlapping
32-token-context windows**, spread across the corresponding split. This is
windowed token perplexity, not an exhaustive full-document WikiText benchmark.
The same target positions are used for every architecture and seed.

Likelihood is reported as mean natural-log NLL per target, bits/token and token
perplexity. Token accuracy is secondary. Do not compare these units directly
with the historical character pilot's BPC or published word-level perplexity.

The additional unigram/bigram/trigram references fit counts on the entire
training stream with add-half smoothing. They are not matched to the neural
models' sampled training-token budget. The five neural conditions form the
budget-matched comparison; n-grams are supplementary references.

Source hashes, exact tokenizer settings and licensing are in
[`data/bpe_full/provenance.json`](data/bpe_full/provenance.json).

## Models

| Condition | Architecture | Trainable parameters |
| --- | --- | ---: |
| Real | 16,384 anatomical neurons, 1,187,999 directed edges | 2,257,055 |
| Shuffled | Same ports, codes and initial weights; target-stub permutation | 2,257,055 |
| Frozen | Same anatomical model; only decoder trained | 1,052,672 |
| GRU | Embedding 128, one hidden layer of 320, untied output | 2,271,104 |
| Transformer | Width 160, three pre-LN blocks, four heads, FF640 | 2,248,096 |

The three fully trainable architectures are within 1% in parameter count.
Frozen-core training intentionally has fewer trainable parameters, although it
stores the same anatomical model as the real condition.

The ordinary models learn token embeddings. The connectome uses fixed bipolar
codes on 192 sampled input neurons and a 256-neuron readout. Thus the ordinary
models are practical language-model references, not isolated tests of wiring.
The real-versus-shuffled comparison is the relevant wiring control.

The Transformer has learned positions, an untied output, GELU and causal
attention. Neither ordinary model uses dropout or pretrained parameters.
The connectome still uses continuous leaky-tanh states, not biological spiking
dynamics, and the selected subgraph remains strongly optic-lobe-biased.

Core-only configuration fields such as `leak`, `initial_gain`, `readout_neurons`
and `edge_chunk` are unused by the ordinary model architectures. Their actual
dimensions are fixed by the model modules and reported parameter counts.

## Fixed training budget

- Seeds: **0, 1 and 2**, five conditions each, 15 runs in total.
- Each run: **8,000 updates**, batch 8, context 32.
- Training targets per run: **2,048,000**.
- Training windows are sampled with replacement, with the same private window
  RNG for paired conditions within a seed.
- This is approximately **0.724 nominal corpus passes per run**, not an
  exhaustive pass. Undertraining is a limitation of the fixed Free Tier budget.
- All positions in each training window contribute next-token cross-entropy.
  Hidden state resets at every training window.
- Float32, AdamW, no weight decay, global gradient clipping at 1.
- Learning rate: `.003` for anatomical conditions; `.001` for GRU/Transformer.
  These rates were declared beforehand, not selected by a score search.
- Save every 250 updates; resume without resetting the optimizer or window RNG.
- Final-update reporting only: no early stopping or validation/test selection.

All five models generate using the same **rolling 32-token context**. Each
prediction recomputes that context from zero state; no recurrent model gets
unbounded history unavailable to the Transformer. The old character path
retains its original stateful-generation default.

## T4 budget measurements

The disposable profiles use one cold forward/backward/AdamW step followed by
20 warm updates on training tokens. Profile weights are never used to initialize
main runs. No held-out loss is scored by the profiler.

| Condition | Warm seconds/update | Peak allocated GPU MiB |
| --- | ---: | ---: |
| Real | 0.184612 | 177.46 |
| Shuffled | 0.207371 | 188.73 |
| Frozen | 0.025541 | 125.48 |
| GRU | 0.004988 | 64.29 |
| Transformer | 0.009819 | 63.69 |

The predicted total for 15 runs is **10,375.94 seconds**, approximately
2 hours 53 minutes of training. Evaluation, checkpoint I/O, transfers and
restoration checks add time. These are estimates from warm measurements, not
the final observed run durations.

GPU figures are PyTorch peak allocations, excluding driver/context memory and
unused allocator reservations. The profiler's process RSS is cumulative within
one process and must not be read as isolated per-model host memory.
Machine-readable evidence is under [`results/bpe-profile/`](results/bpe-profile/).

## Reproduction

Prepare the graph and source files as described in
[AUTOREGRESSIVE.md](AUTOREGRESSIVE.md), then:

```bash
python -m scripts.prepare_bpe_full
python -m scripts.profile_bpe_models
python -m scripts.run_bpe_study --job 0
```

Jobs `0..4` are real, shuffled, frozen, GRU and Transformer for seed 0;
`5..9` repeat seed 1 and `10..14` repeat seed 2. Each job verifies source hashes
and resumes its own existing checkpoint if present. Results are written under
`results/bpe-main/seed-N/CONDITION/`.

Run the shared Torch tests without filtering framework warnings:

```bash
python -m scripts.verify_torch -q
```

The wrapper explicitly enables sparse invariant-check defaults. This resolves
the PyTorch 2.11 warning about an unspecified global checking policy while
retaining the existing per-call opt-out for already-validated cached topology.

For each completed main run, `scripts.verify_ar_resume` works on a separate
verification copy, checks exact restoration and measures subsequent CUDA
continuation differences. CUDA continuation is not assumed to be bitwise
identical. Keep each source checkpoint unchanged and recover it before
releasing a Colab runtime.
