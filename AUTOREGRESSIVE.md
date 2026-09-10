# Connectome-constrained character language model

This experiment asks whether a recurrent network whose trainable connections
follow an actual fly connectome can learn next-character probabilities from text.
It uses maximum likelihood and backpropagation, not the older language-reward
experiment in `LANGUAGE.md`.

## What is anatomical, and what is learned?

Each retained neuron has one continuous scalar state. For character `x`, a fixed
bipolar code `c(x)` enters 192 randomly selected sensory neurons. The recurrence is:

```text
h_next = (1 - leak) * h + leak * tanh(W * (h + c(x)) + bias + c(x))
logits = h_next[readout_neurons] @ readout + output_bias
```

The sensory and readout sets are disjoint. The 256-neuron linear readout has
12,336 parameters for the 48-character alphabet. There is no pretrained language
encoder, learned embedding, attention layer, or direct sensory-to-decoder bypass.
Training updates existing edge weights, neuron biases, and the readout.

The topology is anatomical; the dynamics and training rule are not a faithful
biophysical simulation. Initial weight magnitudes use `log1p` contact counts and
target fan-in normalization. Functional signs are randomly initialized, then
learned without sign constraints; they are not measured neurotransmitter signs.

Sparse recurrence uses cached canonical COO patterns, live edge values, native
PyTorch sparse multiplication, and a custom backward with bounded edge chunks.
Duplicate edges in the control are summed without losing their individual
parameter gradients. No dense neuron-by-neuron matrix is constructed.

## Data and controls

MaleCNS v1.0 includes brain and ventral nerve cord. The full retained annotated
neuronal graph has 166,700 neurons and 25,582,938 directed pairs. Glia and unresolved
objects are excluded; "full" does not mean every segmentation object or every
biological detail of a living fly.

The language pilot uses a nested, induced 16,384-neuron graph with 1,187,999 pairs.
This deterministic hub-BFS subset is strongly optic-lobe-biased, not a whole brain.
Source files, selection details, checksums, and CC-BY-4.0 attribution are in
`data/large_connectome/README.md` and `manifest.json`.

Three matched conditions use seed 0, identical character codes, ports, initial
edge-order weights, readout initialization, training windows, and update budget:

| Condition | Core |
| --- | --- |
| `real` | Train weights on the anatomical graph |
| `shuffled` | Train after permuting target stubs |
| `frozen` | Keep anatomical weights and neuron biases fixed; train the readout |

The shuffled condition preserves directed degree counts but permits parallel
edges and self-loops. Original edge-order weights are retained without subsequent
row renormalization, so incoming strength distributions can change. This is not
an isolated causal test of biological wiring superiority.

WikiText-2 is lowercased and whitespace-normalized. Training uses 250,000
characters; validation uses 65,536. The alphabet is fitted only on training.
The final test uses normalized test characters `[65536:131072]`, excluding the
prefix evaluated in the older RL experiment. Its corpus fingerprint is
`43404b2869c84abf719bff18600c2d463d73c1b41ba41190e6f35692d80d3a4e`.
Text licensing and preprocessing provenance remain under `data/wikitext2/`.

## Fixed pilot protocol

- AdamW, learning rate 0.003, no weight decay, global gradient clip 1.
- Leak 0.5, initial gain 0.9, context 32, batch size 8.
- 4,000 updates per condition: 1,024,000 next-character training targets.
- All positions in each training window contribute to cross-entropy.
- Hidden state resets between training/evaluation windows, not during generation.
- Final-target evaluation uses 1,024 deterministic windows per split.
- Train-fitted unigram, bigram, and trigram baselines use the exact same targets.
- Greedy and sampled continuations each generate 240 characters without future
  text input, from a training-only prompt.
- Evaluation also disables the recurrent operator. Because input reaches the
  disjoint readout through this operator, this ablation removes input propagation
  as well as recurrent communication; it is not a temporal-memory-only ablation.

The scale was selected from synthetic capacity timings, not held-out scores.
The final update budget is fixed; validation and test do not select checkpoints.
A single initialization is an exploratory pilot, not a multi-seed significance
claim. Final measurements and limitations belong in `AR_RESULTS.md`.

## Run on Colab T4

Enable a T4 runtime and place the project at `/content/flyrl-ar`. The delivered
archive includes the prepared pilot graph and corpus. Install the package without
upgrading Colab's existing compatible Torch:

```bash
python -m pip install -e '.[language]'
python -m pip install numpy==2.5.3 pydantic==2.13.5 typer==0.27.2
bash run_ar_colab.sh --output results/ar-fresh
```

The recorded runtime uses Torch 2.11.0+cu128 and NumPy 2.5.3. Keep these versions
when resuming the delivered CUDA checkpoint. CPU execution is also supported
with `--device cpu`, but it is not an interchangeable CUDA-resume runtime.

The delivery includes completed checkpoints under `results/ar-main`. Use a new
output directory for a fresh run, or explicitly resume those checkpoints.
The wrapper accepts additional CLI options, for example:

```bash
bash run_ar_colab.sh --output results/another-seed --seeds 1
bash run_ar_colab.sh --resume --updates 5000
python -m flyrl.autoregressive --help
```

`--updates` is a total target, not an additional count. Progress records go to
stderr; complete JSON reports go to stdout and `report.json`. Checkpoints and
traces are written every 200 updates by the pilot wrapper.

For source-only distributions without prepared graph arrays:

```bash
python -m pip install -e '.[connectome]'
python -m scripts.prepare_large_connectome
python -m scripts.prepare_ar_corpus
```

The large importer downloads about 1.07 GB of pinned upstream data. It is not
needed when the prepared graph is already present. The full 105.8 MB graph is
excluded from Git and ordinary source distributions, but can be reproduced.
`scripts/ar_capacity.py` and the saved capacity JSON document whole-graph
forward/backward/optimizer feasibility separately from the subset language run.

## Checkpoints and verification

Pickle-free NPZ checkpoints contain parameters, Adam moments and step counts,
the private window-sampling RNG, the trace, and graph/corpus/config/runtime
identities. A fully flushed temporary archive replaces the destination atomically.
Existing runs require explicit `--resume`; incompatible checkpoints are rejected.

CPU interrupted/uninterrupted equivalence is regression-tested. CUDA sparse
reductions need not be bitwise deterministic. The verification tool checks exact
saved-state restoration and measures subsequent numerical differences:

```bash
python -m scripts.verify_ar_resume \
  --graph data/large_connectome/malecns_v1_n16384.npz \
  --corpus data/ar_corpus/corpus.npz \
  --run results/ar-main/seed-0/real \
  --output results/ar-resume/real
```

It writes only separate verification copies and verifies that the published
checkpoint's hash remains unchanged.

```bash
uv run --no-sync ruff check flyrl scripts tests
uv run --no-sync basedpyright --pythonversion 3.12 flyrl scripts tests
uv run --no-sync python -m pytest -q
uv build --no-sources
```

Dense-reference derivatives, duplicate edges, causality, frozen controls, actual
CLI execution, checkpoint continuation, and large-graph memory bounds are covered.
`scripts/ar_gpu_checks.py` additionally executes derivative checks on CUDA after
the Colab bootstrap.
