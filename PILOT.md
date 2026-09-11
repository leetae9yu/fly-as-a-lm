# TinyStories connectome pilot

This pilot asks whether a real fruit-fly wiring graph can learn next-token
prediction on short, simple stories. Training and token-aligned neuron
visualization belong to the same pipeline. It does not add a Transformer,
attention mechanism, GRU or pretrained language model around the circuit.

**Execution boundary:** prepare and verify locally first. Colab allocation and
GPU training are not part of this implementation run. A tiny CPU smoke test
checks the software; it is not evidence that the intended 16,384-neuron model
has learned TinyStories.

## Circuit

The intended graph is `data/large_connectome/malecns_v1_n16384.npz`: the existing
16,384-neuron, 1,187,999-edge MaleCNS subset. Its optic-lobe selection bias and
model assumptions remain those in [the graph record](data/large_connectome/README.md).
Smaller existing graph files are for explicit local smoke checks.

```text
token -> learned sensory code -> anatomical recurrence -> linear readout
                                      |
                                      +-> pre-selection state recording
```

The pilot learns the input code on the same disjoint sensory ports used by the
existing model. This is a small vocabulary-by-input-port embedding, not a
vocabulary-by-all-neurons matrix or a separate contextual encoder. Existing
anatomical edges, neuron biases and the readout are trained by next-token
cross-entropy and backpropagation. The topology is fixed. No biological
neurotransmitter identity is inferred from the learned weight signs.

The new trainable-input option is opt-in for historical configurations, so the
published WikiText checkpoints retain their fixed-code behavior. TinyStories
scores must not be compared numerically with the historical WikiText scores as
if the corpus, vocabulary and evaluation protocol were the same.

## Data policy

Use the original `roneneldan/TinyStories` text files, not the separate V2-GPT4
variant. The pinned dataset revision is
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. The
[upstream card](https://huggingface.co/datasets/roneneldan/TinyStories/blob/f54c09fd23315a6f9c86f9dc80f725de7d8f9c64/README.md)
describes synthetic GPT-3.5/GPT-4 stories and specifies CDLA-Sharing-1.0.
This data is not covered by the project's MIT license.

Fit byte-level BPE on selected training stories only. Preserve complete story
boundaries, keep held-out stories out of tokenizer training, and do not train
on windows that cross from one story to another. Record source identity,
selected-story hashes and split boundaries with the corpus. Exact duplicate
filtering does not establish absence of paraphrases or semantic overlap.

The local preparer pools its explicitly supplied files, selects the first
requested number of unique stories and uses a seeded permutation to assign
custom train/validation/test splits. These are **not the official dataset
splits**, and the prefix subset is not a representative random sample of the
full dataset. For the default acquisition path, use only the official training
file as the source pool; keep the official validation file out of training.

## Local preparation and smoke check

Install the project's language extra in a fresh environment:

```bash
uv sync --extra language
```

For the optimized CUDA float32 backward, use `uv sync --extra language --extra cuda`
in a fresh compatible environment. Triton is loaded lazily only by that CUDA
path; CPU execution does not require it. See [measured optimization and
compatibility](GPU_OPTIMIZATION.md). Preserve Colab's existing compatible
CUDA PyTorch/Triton installation rather than replacing it with CPU packages.

In the existing CPU workspace, use `uv run --no-sync` so dependency resolution
does not replace its working PyTorch installation. The language extra now also
includes matplotlib for headless PNG/SVG output and httpx2 for bounded source
acquisition.

First download a small prefix of complete stories from the pinned upstream
training file. This command does not construct a model or start training:

```bash
uv run --no-sync python -m scripts.fetch_tinystories \
  --output artifacts/tinystories-source \
  --split train --stories 32 --max-bytes 262144
```

The fetcher retains the dataset card, license notice and `source.json`, including
the pinned revision and downloaded-prefix hash. It refuses existing output
directories and fails if the byte budget is insufficient for complete stories.

Then prepare the local source and its retained license file:

```bash
uv run --no-sync python -m scripts.prepare_tinystories \
  --raw artifacts/tinystories-source/TinyStories-train.txt \
  --source roneneldan/TinyStories \
  --revision f54c09fd23315a6f9c86f9dc80f725de7d8f9c64 \
  --license-file artifacts/tinystories-source/SOURCE_LICENSE.txt \
  --output artifacts/tinystories-smoke-corpus \
  --train-stories 8 --valid-stories 2 --test-stories 2 \
  --vocab-size 280 --seed 0
```

Preparation writes `corpus.npz`, `tokenizer.json`, `provenance.json` and a retained
`SOURCE_LICENSE.txt`. The local-file path records the caller's source declaration
and consumed-prefix hashes; it cannot authenticate arbitrary local files as
upstream data.

An intentionally small **software check**, not the full pilot:

```bash
uv run --no-sync python -m flyrl.story_pilot \
  --corpus artifacts/tinystories-smoke-corpus/corpus.npz \
  --graph data/large_connectome/malecns_v1_n256.npz \
  --output results/tinystories-smoke \
  --device cpu --cpu-threads 1 \
  --context 8 --batch-size 2 --updates 2 --checkpoint-steps 1 \
  --sample-length 8 --prompt "Once upon a time"
```

The command writes `initial.json`, `report.json`, `checkpoint.npz` and the
activation artifacts described below. It refuses to overwrite an existing run
unless `--resume` is explicit. To continue, repeat the same options, add
`--resume`, and increase `--updates`, which is the **total update target**, not
the number of additional updates. CPU resume requires matching configuration,
graph, corpus and runtime identity.

The intended larger configuration uses the existing 16,384-neuron graph,
4,096 requested BPE vocabulary entries and a 64-token training context. Those
are pilot starting settings, not validated quality targets. A corpus too small
to fit all requested merges can have fewer actual vocabulary entries.
`--no-trainable-codes` provides the fixed-input alternative.
Do not run the default large graph on CPU as a substitute for unavailable Colab.

## Evaluation protocol

Training samples uniformly from legal, full-length within-story windows with
replacement and resets hidden state per window. Stories shorter than
`context + 1` supply no such training windows.

Held-out evaluation counts every next-token pair within each selected story,
including short stories, and excludes first tokens and cross-story transitions.
It resets the recurrent state between stories, but preserves it across
computation chunks within a story. Generation also uses persistent state.
Thus evaluation can use longer context than the truncated training windows;
the report names this policy rather than calling it a matched-window score.
Initial and final validation/test results use the same targets; neither drives
early stopping or checkpoint selection.

## Read the activation output

The run exports:

- `activations.json`: prompt and generated token IDs, raw BPE labels, decoded
  prompt plus continuation, model configuration, update count, graph/corpus/
  parameter fingerprints, context per decision and selected neuron IDs.
- `activations.npz`: **all neurons**, not just the displayed subset. `states`
  has shape `[generated_tokens, neurons]`; columns correspond exactly to
  `node_ids`. It also contains `generated_ids`, selected-token `probabilities`
  and `selected_indices`.
- `activations-001.png` and `.svg`, with additional numbered pages as needed:
  up to 32 generated tokens per page, using the same neuron rows and a fixed
  signed state scale of -1 to 1 across pages.

Each activation row is recorded **after processing the available input and
before selecting the labeled output token**. The newly selected token has not
yet been fed back. Windowed generation resets and replays its declared context;
stateful generation retains its recurrent state. The metadata records which.

The default chart shows the 64 neurons with highest mean absolute state across
the recorded continuation. This is a descriptive selection, not a discovered
language circuit. Rows carry original neuron IDs and input (I), readout (O) or
other (H) port roles. Port roles are engineering choices, not anatomical labels.
The chart is not a spatial brain map. No coordinates or region names are
invented.

Raw byte-BPE labels use visible ASCII escapes in the figure and retain full
labels in JSON. A token can represent only part of a word or UTF-8 character;
decoding tokens one at a time would obscure that identity.

High activation is not attention, causal importance, or proof of language
understanding. Negative state does not identify an inhibitory neuron. Circuit
ablation would be a separate experiment.

The export can also be used with an already loaded anatomical learner:

```python
from pathlib import Path

from flyrl.activations import ActivationOptions, export_activations

path = export_activations(
    learner,
    corpus,
    "Once upon a time",
    Path("results/story-inspection"),
    ActivationOptions(length=32, greedy=True, max_neurons=64),
)
```

The `learner` and `corpus` must come from the same training run. Export is
inference-only and does not change weights, optimizer, progress or training RNG.

## What counts as evidence

Software validation checks correct next-token targets, causal state alignment,
story isolation, training-only tokenization, exact CPU checkpoint continuation
and actual image rendering. A short smoke run is not a model-quality benchmark.

A later authorized training run should report held-out NLL, BPE perplexity and
token accuracy together with generated continuations. Lower teacher-forced loss
alone is not fluent generation. The pilot does not test a biological-wiring
advantage, pure reward-based language acquisition, distillation or RL.
