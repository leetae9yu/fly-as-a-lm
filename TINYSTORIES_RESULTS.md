# TinyStories: first T4 pilot

The anatomical model learned next-token prediction on the selected TinyStories
corpus. It beat the supplementary unigram and bigram references on the same
held-out targets. Free generation acquired recognizable sentence fragments,
but still contains grammatical errors, incoherence and repetition.

This is one seed on a small, nonrepresentative prefix/custom split. It is not a
standard TinyStories benchmark, evidence of language understanding, or a
comparison establishing a biological-wiring advantage.

## Fixed protocol

- Original `roneneldan/TinyStories` training-file revision
  `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`; CDLA-Sharing-1.0.
- Fetched 1,100 complete stories, then selected the first 1,032 unique stories
  and split by seeded permutation into 1,000 train / 16 validation / 16 test.
  Exact duplicate count removed: 0. These are not the official splits.
- Train-only byte BPE: 4,096 tokens. Split sizes:
  230,745 / 4,098 / 3,155 tokens.
- Existing MaleCNS subset: 16,384 neurons, 1,187,999 directed edges.
  Continuous sparse anatomical recurrence, learnable sensory codes, linear
  readout. No Transformer, GRU, attention, distillation or RL.
- 3,043,487 trainable parameters; seed 0; context 64; batch 8; AdamW
  learning rate .003; gradient clipping 1; weight decay 0.
- Fixed 1,000 updates, 512,000 training targets, checkpoints every 100 updates.
  This is about 2.22 nominal training-token passes with replacement, not
  exhaustive epochs. No held-out-driven early stopping or budget changes.
- Training windows remain within stories and reset their state. Evaluation
  scores every within-story next-token pair with full causal recurrent state,
  resetting only between stories. The first token of each story is unscored.
- Stateful generation: 64 tokens from `Once upon a time`; both greedy and
  sampled output recorded. Activation export corresponds to greedy output.

## Held-out results

Lower NLL and perplexity are better. Accuracy is next-BPE-token accuracy.

| Split | Targets | Initial NLL | Final NLL | Initial PPL | Final PPL | Final accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 4,082 | 8.3178 | 4.1986 | 4,096.06 | 66.59 | 26.11% |
| Test | 3,139 | 8.3178 | 4.0603 | 4,096.25 | **57.99** | **28.29%** |

Supplementary references use add-half smoothing and all selected training
stories, with no cross-story transitions. They see the corpus once, whereas
the neural model samples windows with replacement. They are descriptive
references, not matched-exposure architecture controls.

| Test predictor | NLL | Perplexity | Accuracy |
| --- | ---: | ---: | ---: |
| Unigram | 6.0071 | 406.31 | 7.14% |
| Bigram | 5.2508 | 190.73 | 25.23% |
| Anatomical model | **4.0603** | **57.99** | **28.29%** |

These BPE perplexities must not be directly compared with the older WikiText
perplexities: corpus, tokenizer and evaluation context policy differ.

## Generated continuations

Full greedy output, including the prompt:

```text
Once upon a time, there was a little girl named Lily. She loved to the kitchen and said, "I'm sorry, I can't you want to play with me. You can play with your toys."
Lily and Ben said, "I'm sorry, I can't get it."
Lily and Ben said, "I
```

Full sampled output, including the prompt:

```text
Once upon a time with there. Lila wasike who was mean. He bees and someone dry fall.
After wa opened his new friend. They go to play with theure and the words and hugged each other. They think waved their making some saladseshood,. She says, "No, I you can me," share him
```

The natural opening alone is not evidence of coherent storytelling. The later
errors and repetition are part of the result, not omitted samples.

## Neuron activity

Both PNG pages and their SVG counterparts were recovered and visually inspected.
All 16,384 neuron states for all 64 generated positions are preserved in the
NPZ; original neuron IDs map every column. Recording occurs before the selected
token is fed back.

The default chart ranks 64 neurons by mean absolute activation. In this run,
94.14% of displayed cells have absolute state at least .99, so those rows look
nearly constant and saturated. This is a selection effect, not evidence that
the whole circuit is constant: across the full recording, 8.05% of cells meet
that threshold, 16,226 neurons have state standard deviation above .01, and
7,818 above .1. Median neuron standard deviation is .0936.

These are model-state observations on one generated continuation, not firing
rates, attention weights or causal language circuits. A supplementary
variation-ranked view would be more informative about token-to-token changes
than the current magnitude-ranked chart.

## Runtime and recovery

- Actual Tesla T4, Python 3.13.15, Torch 2.11.0+cu128, CUDA 12.8.
- Training including periodic checkpoint writes: **417.18 seconds** (6m57s).
- Initial evaluation through final generation/figure export: 434.21 seconds;
  subsequent checkpoint verification and packaging are excluded.
- Peak allocated CUDA tensors during that run: **271.06 MiB**. This excludes
  the CUDA context, allocator reservation and other process/driver memory.
- Remote preflight: 22 regression tests and two actual CUDA sparse
  forward/backward comparisons against dense references passed.
- The preinstalled Matplotlib triggered a pyparsing deprecation warning under
  strict tests. Pinning the locally verified Matplotlib 3.11.1 resolved it;
  warnings were not suppressed.
- An immutable update-500 checkpoint was downloaded while training continued.
- Final checkpoint loaded into two fresh GPU models with matching parameters,
  Adam moments, RNG and trace; stored greedy and sampled generations reproduced.
  This run did not test bitwise equality of additional resumed CUDA updates.
- Downloaded ZIP CRC, checkpoint SHA256, protocol/report/checkpoint identities,
  finite arrays, original neuron IDs and causal activation contexts verified.

## Artifacts and reproduction

The local command and model documentation are in [PILOT.md](PILOT.md).
This run's artifacts are retained locally:

- `artifacts/tinystories-t4-input.zip`: frozen source distribution, graphs,
  corpus, worker and the predeclared `protocol.json`.
- `artifacts/tinystories-t4-source/` and `artifacts/tinystories-t4-corpus/`:
  source/license/hash records and prepared BPE.
- `artifacts/tinystories-t4-midpoint-checkpoint.npz`: external update-500 backup.
- `artifacts/tinystories-t4-results.zip`: complete downloaded run.
- `results/tinystories-t4-pilot/`: report, checkpoint, raw activations, two
  heatmap pages, runtime verification, environment and count references.

Corpus fingerprint:
`fd9d30ce06ef753feb2160370d40bceebd41f126aaf6be58d9d6ed42cda45575`.
Final checkpoint SHA256:
`e0ad8077ceabf61875e4cf38faffc95e781c1535f6688ed544c1269441a4b99a`.
Result ZIP SHA256:
`479a10ac7c56386953e97912035114229abb861d458e3cfce8ae167a14420b97`.

The GPU was released after recovery; a read-only session check returned no
active sessions. No larger run was launched.

## Expansion assessment

The result justifies another language-learning experiment, but not a claim that
more neurons will resolve the observed errors. The current subset contains
15,355 optic-lobe-intrinsic neurons (93.72%) and only 55 central-brain-intrinsic
neurons. The source annotation has 32,164 `cb_intrinsic` neurons overall.

A central-brain-focused graph is a useful next candidate, preferably with a
size-controlled comparison so regional selection is not confused with scaling.
It should not be called a single established "cognition circuit." Input/output
pathways and discarded boundary connections need an explicit selection policy.

The existing full retained graph has 166,700 neurons and 25,582,938 edges
(21.53 times this run's edges), and includes the ventral nerve cord as well as
the brain. It is not simply "whole brain." Enlarged training was not performed
or timed here; these options require a separate run specification.
