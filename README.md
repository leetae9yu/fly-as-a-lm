# FlyRL: connectome reward-learning experiments

For the **T4 GPU natural-text experiment**, see [LANGUAGE.md](LANGUAGE.md) and
run `bash run_language_colab.sh`. The sections below document the original
CPU binary-task prototype.

A small CPU-only prototype for the research question: can a real Drosophila
connectivity pattern adapt through local reward signals, and how does its
behavior compare with rewired and non-learning controls?

This is an experimental starting point, not a language model or a biologically
validated fly simulation. The original project document supplies the research
philosophy; implementation choices remain hypotheses to test.

## Run locally

Python 3.11 or later and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked
uv run python -m scripts.prepare_data
uv run python -m pytest -q tests/test_connectome.py tests/test_learning.py \
  tests/test_experiment.py tests/test_preparation.py
uv run python -m flyrl \
  --graph data/larva_left_mb.npz \
  --output results/first \
  --episodes 1500 --seeds 0,1,2 --delay 3 --eval-trials 512
```

Continue to a **total** of 3000 episodes per run:

```bash
uv run python -m flyrl \
  --graph data/larva_left_mb.npz \
  --output results/first \
  --episodes 3000 --seeds 0,1,2 --delay 3 --eval-trials 512 --resume
```

There are 18 runs: two tasks, three controls, and three seeds. A new run refuses
to overwrite existing checkpoints without `--resume`. Resumption requires the
same graph, configuration, Python version, and NumPy version. The episode
budget can increase; a smaller target is rejected.

## Run on Colab Free through the CLI

The CLI must already be installed and authenticated. No paid accelerator,
compute-unit purchase, or GPU is requested.

```bash
bash run_colab.sh
```

The script creates a CPU session, uploads the source and pinned data, installs
the tested dependency versions, runs the tests, then runs:

1. 1000 episodes per task/control/seed.
2. The same checkpoints resumed to 1500 episodes.
3. An independent, uninterrupted 1500-episode run.

It compares all 18 final checkpoint files and the summary byte for byte.
Results download to `artifacts/flyrl-results.zip`, and the owned runtime is
released on exit. `FLYRL_SESSION` can set a specific session name.

For a longer experiment, modify the explicit budgets in
`scripts/colab_run.py`. The small default is a verification run, not a
hyperparameter search. Runtime allocation and interruption remain subject to
Colab Free availability.

**Colab storage is temporary.** Saving to `/content` allows recovery from an
interrupted process on a surviving VM, not from destruction of the VM.
Download checkpoint files before releasing a runtime. An explicit CLI example:

```bash
colab download -s YOUR_SESSION \
  /content/flyrl/results/colab-resumed/association-real-0.json \
  association-real-0.json
```

## Data and biological assumptions

The bundled dataset is the **left larval mushroom-body connectome**, not an
adult fly brain: 209 neurons and 7425 directed neuron-pair connections.

- Original study: [Eichler et al., Nature (2017)](https://doi.org/10.1038/nature23455).
- Data distribution: [graspologic's Drosophila dataset](https://github.com/graspologic-org/graspologic/tree/ccf1458bd73b25cb1e7a779c098a54943647bfa1/graspologic/datasets/drosophila).
- Upstream commit and SHA-256 checksums are in `data/provenance.json`.
- `data/raw` retains the unmodified source files. Preparation rejects a checksum
  mismatch rather than silently using different data.
- Rows are sources and columns are targets. All nonzero entries and observed
  self-connections are retained. Initial strengths derive from `log1p(count)`.
- Source labels are retained in row IDs, but are not original skeleton IDs.
- No neurotransmitter annotations are supplied by these files. Positive
  synapse signs are a **model assumption**, not biological evidence.
- The upstream distribution's license is in `data/LICENSE.graspologic.txt`.

## Learning rule and tasks

The first neuron model is a discrete-time stochastic binary recurrent network,
not LIF. This makes the local credit signal explicit:

```text
p_j(t+1) = sigmoid(b + sum_i w_ij * z_i(t))
z_j(t+1) ~ Bernoulli(p_j(t+1))
e_ij += z_i(t) * (z_j(t+1) - p_j(t+1))
w_ij += learning_rate * (reward - previous_baseline) * e_ij
```

Weight updates are projected onto the original signs and magnitude bounds.
Incoming synapses to the clamped sensory neuron do not receive a learning
signal. No autograd, backpropagation through time, learned text encoder,
learned output decoder, or language-model judge is used.

The eligibility rule is a local score-function estimator, **not standard
pair-based STDP**. Reward is broadcast as a scalar teaching signal; anatomical
dopaminergic neurons are not simulated. Neural state and eligibility reset at
trial boundaries.

- **Association:** a binary cue is presented for two simulation steps; the
  final binary state of a fixed output neuron is the action. Reward is 1 when
  action equals cue, otherwise 0.
- **Memory:** a cue is followed by identical zero-input steps. The final action
  must match the initial cue. With `--delay 3`, the stimulus has five steps:
  one cue and four zero-input steps.

Input and output neurons are selected once per seed on the real graph and
shared across its controls. These are artificial task ports, not claims about
the anatomical sensory or motor identity of those cells.

## Interpretation

The controls are real topology with learning, directed degree-preserving
rewiring with learning, and real topology with frozen weights. Rewiring uses a
bounded number of valid edge swaps. It is not a guarantee of uniform graph
sampling or a mixing-time result. Its accepted-swap count is retained.

Weights are normalized by their source neuron's outgoing strength, giving each
source a fixed initial weight budget. Because rewiring retains source indices
and edge-associated raw weights, both learning controls start with identical
edge-ordered weight values.

**The topology comparison has a known task-accessibility confound.** Ports are
chosen on an observed real edge; shuffling may remove that direct connection.
`direct_edge_present` records this for every run, and `topology_caveat` is
included in the summary. A real-graph advantage in this first setup cannot be
attributed to biological topology alone. A follow-up should match port
accessibility or test ports independently of either topology.

Evaluation uses balanced binary cues and a separate fixed random stream.
It does not modify training weights, rewards, or randomness. Here, "held out"
means new stochastic trials, **not unseen sentences or compositional
generalization**. Chance and the best constant action both score 50%.

`summary.json` contains initial/final accuracy, per-cue accuracy, weight change,
settings, selected ports, and per-seed aggregates. Checkpoints also retain the
reward history and complete resumable state. Seed standard deviations are
descriptive; three seeds are not enough for a strong claim of topological
superiority.

Treat a failed memory result as a result to investigate, not a software failure
or proof that biological flies lack memory. The first tasks deliberately avoid
language, full-brain scaling, and 3D visualization.

## Files and checks

- `flyrl/connectome.py`: sparse graph input and rewiring.
- `flyrl/learning.py`: stochastic neural dynamics and local updates.
- `flyrl/experiment.py`: paired control runs and summaries.
- `flyrl/checkpoint.py`: atomic, validated JSON state.
- `scripts/prepare_data.py`: checksum-pinned data preparation.
- `scripts/colab_run.py`: actual remote execution and resume verification.

```bash
uv run ruff check flyrl scripts tests
uv run basedpyright flyrl scripts tests
uv run python -m pytest -q
uv build
bash -n run_colab.sh
```
