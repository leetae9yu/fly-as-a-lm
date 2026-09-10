# Reward-trained character prediction on a T4

This experiment moves from binary toy tasks to next-character prediction on
natural text. It trains a connectome-constrained stochastic network by rewarding
the character it actually sampled. It does not use supervised backpropagation,
a learned text encoder/decoder, a pretrained language model, or an LLM judge.

The first measured run collapsed to the space-frequency baseline rather than
useful contextual prediction. See [LANGUAGE_RESULTS.md](LANGUAGE_RESULTS.md)
for the complete result, generated samples, and verification evidence.

## Run on Colab Free

With the authenticated `colab` CLI already installed:

```bash
bash run_language_colab.sh
```

The script explicitly requests `--gpu T4`. It never purchases compute units or
requests an A100. Allocation depends on current free-tier availability; a
requested CUDA device is never silently replaced with CPU computation.

Colab's existing CUDA-enabled PyTorch is retained. Do not install a CPU-only
PyTorch wheel into this runtime. The runner resets the kernel after installing
the other dependencies and sets `CUBLAS_WORKSPACE_CONFIG` before CUDA matmul.

The prescribed run is:

| Setting | Value |
|---|---|
| Connectome | 209 larval mushroom-body neurons; 7425 directed edges |
| Corpus | Tokenized WikiText-2; separate official splits |
| Prefix sizes | 250,000 training; 65,536 validation; 65,536 test characters |
| Vocabulary | 48 characters, fitted on training text only |
| Context | 16 teacher-forced characters |
| Action | One sampled next character |
| Training | 1000 updates of 128 windows per condition |
| Conditions | Real, degree-preserving rewiring, and frozen real |
| Seeds | 0, 1, 2 |
| Evaluation | 2048 non-overlapping windows spread over each split |
| Checkpoints | Every 100 updates |
| Selection | Prespecified final update, not the best test result |

The main run is split at update 500, then resumed to 1000. A separate
uninterrupted 1000-update run for seed zero checks all three controls. It compares
checkpoint array/metadata contents, final metrics, rewards, and generated
continuations. ZIP container timestamps are not treated as model state.

The script downloads `artifacts/flyrl-language-results.zip` before releasing the
runtime. As with the original prototype, `/content` is temporary: for a custom
long run, export checkpoints before the VM is destroyed. GPU checkpoint reuse
requires the matching graph, corpus, configuration, PyTorch version, and device.

## Custom execution

Within the uploaded project on a CUDA-equipped runtime:

```bash
python -m flyrl.language \
  --graph data/larva_left_mb.npz \
  --corpus data/wikitext2/corpus.npz \
  --output results/my-language-run \
  --device cuda --updates 1000 --batch-size 128 --context 16 \
  --seeds 0,1,2 --eval-windows 2048 --learning-rate 0.1
```

Use `--resume --updates 2000` with the same other settings to continue to
**2000 total updates**, not to add 2000. Changing a context, seed, vocabulary,
batch size, or learning rate is a new experiment, not compatible resumption.
For a new context or evaluation-window count, recompute n-gram scores at the
same target positions instead of comparing against the default baseline file.

The `language` project extra declares the PyTorch requirement. Local CPU
development can use a CPU-only PyTorch wheel; the real experiment requires
CUDA and was designed for the preinstalled Colab GPU distribution.
After installing the corresponding dependencies:

```bash
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check flyrl scripts tests
uv run --no-sync basedpyright flyrl scripts tests
uv build
```

## Model and local credit assignment

The same anatomical mask and initial positive-sign assumption from the first
prototype are retained. Source-normalized initial weight values match across
the real and shuffled graphs. The implementation uses **dense masked float32
matrix kernels on the sparse anatomical graph**, not a sparse-spiking simulator.
Dense kernels are a practical GPU choice for this small graph.

A seeded permutation of neuron indices assigns disjoint fixed input and output
neurons, one of each per character. The permutation never reads connectivity.
The assignments are identical across controls, removing the earlier experiment's
selection of ports on a known real edge. These assigned roles are artificial,
not anatomical labels.

For intermediate context steps, other neurons sample Bernoulli states. Each
allowed synapse accumulates a presynaptic-state times postsynaptic-score term:

```text
hidden probability = sigmoid(previous_state @ weights)
eligibility += previous_state * (sampled_hidden_state - hidden_probability)
```

At the final step, fixed output neurons form a categorical policy. The final
local term uses the **sampled action**, not the correct label:

```text
policy = softmax(final_output_potentials)
action ~ Categorical(policy)
eligibility[:, output] += previous_state * (one_hot(action) - policy)
reward = 1 if action == next_character else 0
weight_update = learning_rate * mean((reward - previous_baseline) * eligibility)
```

Synapse updates preserve the original mask and signs. Incoming sensory weights
are not plastic. No biases are trained. The reward baseline is updated after
the weight update. These are local likelihood-ratio scores, not pair-based STDP
or a claim of faithful biological dopamine signaling.

Each window starts from a reset neural state; recurrence occurs inside the
16-character window. The target is never supplied to the rollout. Training
windows can overlap within the training split but cannot cross into another
split. Eligibility storage is proportional to batch size times neuron count
squared, not multiplied by the context length.

## Evaluation and interpretation

Evaluation uses independent seeded randomness and does not change training
state. It reports:

- Greedy next-character accuracy, conditional on a sampled hidden trajectory.
- Expected reward, integrating the categorical action probabilities.
- Actual sampled-action accuracy.
- Conditional negative log probability and bits per character.
- Short autoregressive continuations using the model's own generated input.

The NLL/BPC is conditional on sampled hidden trajectories. It is not an exact
marginal likelihood over all hidden trajectories or standard word perplexity.
Accuracy reward is not a proper probability-scoring rule: optimizing it may
make a policy less calibrated, so BPC can worsen while reward improves.

The comparisons in `data/wikitext2/baselines.json` use the same target positions:
uniform random accuracy, most-frequent-character/unigram, bigram, and trigram
models fitted only to training text. Beating uniform random is not sufficient.
Beating the most frequent character is the first indication of useful
conditioning; stronger sequence or grammatical claims need additional evidence.

No test-dependent tuning or checkpoint selection is performed. Three graph/port
seeds and one text prefix are a pilot, not a broad language benchmark or proof
that biological topology is advantageous. Real text in the environment does
not by itself mean the network acquired grammar or semantic understanding.

See `data/wikitext2/README.md` for text provenance, tokenization artifacts,
normalization, and licensing. The original binary experiment remains available
through `python -m flyrl` and `run_colab.sh`.
